#!/usr/bin/env python3
# Copyright   2026  (author: Ibrahim Almajai)
# Apache 2.0

"""
Interactive lipreading (VSR) demo for the GRID conformer_ctc2 recipe.

Pipeline (all stages reused from the recipe):

    video ->  dlib face + 68 landmarks        (local/compute_avhubert_grid.py)
          ->  mouth ROI crop, 88x88 grey
          ->  AV-HuBERT feature extract        (extract_finetune, output_layer=L)
          ->  Conformer-CTC forward
          ->  CTC greedy decode -> text

Two modes:

  (1) CLI -- print the recognised text for one video:

    ./conformer_ctc2/demo.py \
        --checkpoint conformer_ctc2/exp32/pretrained.pt \
        --tokens data/lang_bpe_58/tokens.txt \
        --method 1best \
        --HLG data/lang_bpe_58/HLG.pt \
        --words-file data/lang_bpe_58/words.txt \
        --avhubert-ckpt download/avhubert-ckpts/base_vox_iter5.pt \
        grid-corpus/s33/pgwh8n.mpg

  (2) Gradio web UI -- upload a clip, see the mouth ROI strip + text:

    ./conformer_ctc2/demo.py --ui \
        --checkpoint conformer_ctc2/exp32/pretrained.pt \
        --tokens data/lang_bpe_58/tokens.txt \
        --method 1best \
        --HLG data/lang_bpe_58/HLG.pt \
        --words-file data/lang_bpe_58/words.txt \
        --avhubert-ckpt download/avhubert-ckpts/base_vox_iter5.pt

Decoding: --method ctc-greedy (default, needs only --tokens) or --method 1best
for lexicon-constrained output (add --HLG and --words-file). Works in both CLI
and UI modes. Defaults (encoder-dim 128, 6 enc / 3 dec layers, layer 9) match
the exp32 model; override --encoder-dim/--num-encoder-layers/--num-decoder-layers
for a differently-shaped checkpoint.

Clips at other frame rates (e.g. 30/60 fps phone video) are resampled to 25 fps
-- the GRID / AV-HuBERT rate -- with ffmpeg before feature extraction, so lip
motion reaches the model at the speed it was trained on.
"""

import argparse
import logging
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

# Bundled py3.8-compatible MediaPipe (the env's own build is broken on 3.8),
# used as a robust fallback landmarker. It must be imported BEFORE anything
# that pulls in the env's protobuf 5.x (fairseq, tensorboard): MediaPipe
# needs the bundled protobuf<4 and whichever protobuf loads first wins.
# Nothing else in the demo process needs protobuf 5.
MEDIAPIPE_DIR = Path(__file__).resolve().parent.parent / "download/mediapipe-py38"
try:
    sys.path.insert(0, str(MEDIAPIPE_DIR))
    import mediapipe as _mediapipe  # noqa: E402
except Exception:  # missing dir/deps: demo runs without the fallback
    _mediapipe = None

from conformer import Conformer
from cmn import apply_cmn
from icefall.checkpoint import load_checkpoint
from icefall.utils import str2bool

# `compute_avhubert_grid` lives in the recipe's local/ dir, which is not on
# sys.path when running conformer_ctc2/demo.py; add it before importing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local"))
from compute_avhubert_grid import (  # noqa: E402
    load_globals_avbubert,
    load_globals_dlib,
    _detect_landmarks,
    _extract_mouth_frames,
)

# Mouth-ROI extraction constants (same values used in feature precomputation).
ROI_SIZE = (88, 88)
MOUTH_LEFT, MOUTH_RIGHT = 48, 54  # dlib 68-point landmark indices
MOUTH_W = MOUTH_H = 64
BLANK_ID = 0  # icefall CTC blank
TARGET_FPS = 25.0  # GRID / AV-HuBERT frame rate; other rates are resampled
# Frames taller than this are downscaled for face *detection* only (dlib's
# HOG scan is ~linear in pixels; phone videos are 1080p+). Landmarks and the
# mouth crop still use full resolution.
DETECT_HEIGHT = 480
# Accepted clip length. Below the minimum no GRID-style sentence fits; above
# the maximum, per-frame landmarking would tie up the demo for too long.
# The tolerance keeps nominal-length clips from being rejected over container
# metadata rounding.
MIN_DURATION_S = 2.0
MAX_DURATION_S = 15.0          # uploaded files
MAX_DURATION_WEBCAM_S = 20.0  # webcam recordings get a little more headroom
DURATION_TOL_S = 0.1

# Head-size normalization. GRID has near-uniform framing: its fixed 64px
# mouth crop corresponds to a median interocular distance (IOD) of 48.4px,
# measured over the corpus landmarks (33 speakers x 40 clips; per-speaker
# medians span 40-56px, so training itself saw ~+-16% mouth-scale variance).
# For arbitrary videos the face can be any size, so we scale the crop to the
# measured IOD (crop = CROP_PER_IOD * IOD) before resizing to ROI_SIZE,
# keeping the mouth-to-head ratio consistent with GRID; for a median GRID
# face this reproduces the training 64px crop. Validated by a crop-scale
# sweep (1.15-1.55 x IOD): WER varies by only ~1 point over that range on
# Lombard GRID, and this value edged out the previous mis-measured reference
# (50.6 -> crops ~4.5% small) on both GRID and Lombard. IOD is a valid scale
# proxy cross-corpus: mouth-width/IOD is 0.807 on GRID vs 0.801 on Lombard.
GRID_IOD_REF = 48.4
CROP_PER_IOD = MOUTH_W / GRID_IOD_REF  # ~1.322


def _degrade_roi(roi: np.ndarray, noise_sigma: float,
                 blur_sigma: float) -> np.ndarray:
    """Optionally blur then add Gaussian noise to ROI frames (T,H,W uint8),
    for the UI's robustness playground. Blur first: camera blur precedes
    sensor noise in a real pipeline."""
    import cv2

    out = roi.astype(np.float32)
    if blur_sigma > 0:
        out = np.stack(
            [cv2.GaussianBlur(f, (0, 0), blur_sigma) for f in out]
        )
    if noise_sigma > 0:
        rng = np.random.default_rng()
        out = out + rng.normal(0.0, noise_sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_mediapipe_fallback():
    """Build a fallback landmarker from MediaPipe Face Detection (BlazeFace,
    short-range): fast (~17 ms/frame CPU) and robust to glasses, tight
    framing and mild pose, exactly where dlib's HOG detector fails.

    Returns a callable (BGR frame) -> (68, 2) int array or None. Only the
    points the pipeline consumes are meaningful: the mouth-corner midpoint
    (indices 48/54, set so their mean is MediaPipe's mouth center; measured
    offset vs dlib is ~0.01-0.07 IOD with no systematic bias) and the eye
    blocks 36:42 / 42:48 (set to the eye keypoints, so the interocular
    distance for head-size normalization stays valid, within ~10% of dlib's).
    """
    import cv2

    if _mediapipe is None:
        raise RuntimeError(f"bundled mediapipe not importable ({MEDIAPIPE_DIR})")
    mp = _mediapipe

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    KP = mp.solutions.face_detection.FaceKeyPoint

    def fallback(image_bgr):
        h, w = image_bgr.shape[:2]
        result = detector.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not result.detections:
            return None
        kps = result.detections[0].location_data.relative_keypoints

        def px(k):
            return (kps[k].x * w, kps[k].y * h)

        lm = np.full((68, 2), px(KP.MOUTH_CENTER), dtype=float)
        lm[36:42] = px(KP.RIGHT_EYE)  # subject's right eye = dlib 36-41
        lm[42:48] = px(KP.LEFT_EYE)   # subject's left eye = dlib 42-47
        return np.round(lm).astype(int)

    return fallback


def _median_iod(landmarks: np.ndarray) -> float:
    """Median interocular distance (px) across frames from dlib 68 landmarks."""
    lm = landmarks.astype(float)
    reye = lm[:, 36:42].mean(axis=1)  # right-eye centre per frame
    leye = lm[:, 42:48].mean(axis=1)  # left-eye centre per frame
    return float(np.median(np.linalg.norm(leye - reye, axis=1)))

# Shown in the UI so users know what a valid GRID sentence looks like.
GRID_GRAMMAR_MD = """
Every GRID sentence follows a fixed 6-word grammar:

**command · colour · preposition · letter · digit · adverb**

| slot | choices |
| --- | --- |
| command | bin, lay, place, set |
| colour | blue, green, red, white |
| preposition | at, by, in, with |
| letter | a–z (except **w**) |
| digit | zero, one, two, three, four, five, six, seven, eight, nine |
| adverb | again, now, please, soon |

Example: **place green with h eight now**
"""

# Per-visitor (IP) usage quotas for the UI, hourly and daily, counted over
# all submissions regardless of source (upload or webcam). Kept in memory
# only, so a server restart clears the counters. IP-based counting is a
# courtesy limit for volunteers, not a security boundary (shared NATs pool
# their quota).
QUOTA_HOUR_S = 3600
QUOTA_DAY_S = 24 * 3600

_usage_log = {}  # ip -> submission timestamps within the last day



def _client_ip(request) -> str:
    """Real visitor IP. Behind the Cloudflare tunnel the TCP peer is the
    cloudflared container, so trust Cf-Connecting-IP (set by Cloudflare's
    edge), then X-Forwarded-For, before falling back to the socket peer."""
    if request is None:
        return "unknown"
    headers = getattr(request, "headers", None) or {}
    ip = headers.get("cf-connecting-ip", "")
    if not ip:
        ip = headers.get("x-forwarded-for", "").split(",")[0].strip()
    if not ip and getattr(request, "client", None):
        ip = request.client.host
    return ip or "unknown"


def _is_local_ip(ip: str) -> bool:
    """True for loopback/private addresses. Public traffic always arrives
    with a Cloudflare-set public IP, so a private address means the request
    came from the host machine itself (the operator): exempt from quotas."""
    import ipaddress

    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private


def _quota_exceeded(ip: str, hour_limit: int, day_limit: int):
    """Record one use unless a limit is hit. Returns None when allowed,
    otherwise the name of the exceeded window ('hourly' or 'daily')."""
    now = time.time()
    recent = [t for t in _usage_log.get(ip, []) if now - t < QUOTA_DAY_S]
    if len(recent) >= day_limit:
        _usage_log[ip] = recent
        return "daily"
    if sum(1 for t in recent if now - t < QUOTA_HOUR_S) >= hour_limit:
        _usage_log[ip] = recent
        return "hourly"
    recent.append(now)
    _usage_log[ip] = recent
    return None


# Attribution shown in the UI footer (GRID is CC BY 4.0 -> attribution due).
ACKNOWLEDGEMENTS_MD = """
---
**Acknowledgements**: This research demo uses the
[GRID audiovisual sentence corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/)
(Cooke, Barker, Cunningham & Shao, 2006; CC BY 4.0) and Meta's
[AV-HuBERT](https://github.com/facebookresearch/av_hubert) visual front-end
(Shi et al., 2022; non-commercial research license), and is built with
[icefall](https://github.com/k2-fsa/icefall) / k2 and
[dlib](http://dlib.net/) facial landmarks.
"""

def _grammar_fst_svg() -> str:
    """The GRID grammar drawn as a linear ("sausage") FST: 7 states in a row,
    one bundle of parallel word arcs per slot. Inline SVG that scales to the
    container width and uses currentColor, so it follows the UI theme. The
    25-letter and 10-digit slots are elided with a dashed '...' arc; the full
    inventories are in the table below the drawing."""
    slots = [
        ("command", ["bin", "lay", "place", "set"]),
        ("colour", ["blue", "green", "red", "white"]),
        ("preposition", ["at", "by", "in", "with"]),
        ("letter (no w)", ["a", "b", "…", "z"]),
        ("digit", ["zero", "one", "…", "nine"]),
        ("adverb", ["again", "now", "please", "soon"]),
    ]
    seg_w, x0, y, r = 170, 40, 170, 14
    apexes = [-84, -28, 28, 84]  # vertical peak of each parallel arc
    parts = []
    for i, (title, words) in enumerate(slots):
        x1, x2 = x0 + i * seg_w, x0 + (i + 1) * seg_w
        cx = (x1 + x2) // 2
        parts.append(
            f'<text x="{cx}" y="28" text-anchor="middle" font-weight="bold" '
            f'font-size="20" fill="currentColor">{title}</text>'
        )
        for a, w in zip(apexes, words):
            dash = ' stroke-dasharray="4 4"' if w == "…" else ""
            parts.append(
                f'<path d="M {x1 + r} {y} Q {cx} {y + 2 * a} {x2 - r} {y}" '
                f'fill="none" stroke="currentColor" stroke-opacity="0.45" '
                f'marker-end="url(#arr)"{dash}/>'
            )
            ty = y + a + (-10 if a < 0 else 20)
            parts.append(
                f'<text x="{cx}" y="{ty}" text-anchor="middle" '
                f'font-size="19" fill="currentColor">{w}</text>'
            )
    for i in range(len(slots) + 1):
        x = x0 + i * seg_w
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" '
            f'stroke="currentColor" stroke-width="2"/>'
        )
        if i == len(slots):  # final state: double circle
            parts.append(
                f'<circle cx="{x}" cy="{y}" r="{r - 4}" fill="none" '
                f'stroke="currentColor" stroke-width="2"/>'
            )
        parts.append(
            f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-size="14" '
            f'fill="currentColor">{i}</text>'
        )
    return (
        '<svg viewBox="0 0 1100 300" role="img" '
        'aria-label="GRID grammar as a linear FST" '
        'style="width:100%;height:auto" xmlns="http://www.w3.org/2000/svg">'
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#888"/></marker></defs>'
        + "".join(parts) + "</svg>"
    )


# HLG lattice-decoding hyper-parameters (from decode.py get_params()).
SEARCH_BEAM = 20
OUTPUT_BEAM = 4
MIN_ACTIVE_STATES = 30
MAX_ACTIVE_STATES = 10000
# Word confidences come from a second, wider lattice built on
# temperature-flattened posteriors: CTC output is too peaky and the tight
# decoding beam prunes competitors away, so posteriors on the decoding
# lattice score even wrong words ~1.0. Values chosen by AUC (0.91 vs 0.72
# for the decoding lattice) on a held-out unseen-speaker sweep.
CONF_OUTPUT_BEAM = 10
CONF_TEMPERATURE = 2.0
# Out-of-domain detection (1best only), two independent signals; either
# firing hides the result. Calibrated on 15 real in-domain clips (plain +
# Lombard GRID, several speakers, some with genuine recognition errors) and
# 3 real non-GRID clips -- small samples, especially on the OOD side, so
# treat both thresholds as provisional.
# TODO: refine on more real non-GRID recordings ([[grid-demo-ood-calibration]]).
#
# Signal 1: mean non-blank emission log-prob per frame of the best path,
# i.e. how well the lip frames match the decoded GRID sentence. In-domain
# GRID speech sits near 0 (~-0.03, worst ~-0.17); non-GRID motion CAN drop
# well below, but doesn't always -- this signal alone missed 2 of 3 known
# OOD clips (they scored -0.115 and -0.141, both better than the in-domain
# worst case).
OOD_HIDE_THRESHOLD = -0.50
# Signal 2: std of the per-frame max-softmax (top-1 probability, taken
# straight off nnet_output -- decode-independent, unlike signal 1) over
# non-blank frames. Genuine GRID speech gives temporally stable per-frame
# confidence even when the model picks the wrong word; unfamiliar mouth
# motion makes it swing frame to frame.
#
# This signal reads off the output distribution, so it does NOT transfer
# across models with different vocabularies. Under the BPE-58 system: 15
# in-domain clips 0.004-0.142, 3 OOD clips 0.157-0.208, threshold 0.15 at
# the midpoint. Under the phone system the in-domain distribution sits
# higher -- 200 GRID test clips give mean 0.089, p99 0.161, max 0.177 --
# and 0.15 would hide 4% of legitimate results.
#
# Retuned from the in-domain distribution alone (200 clips, versus 15
# before), placed above the observed maximum so that correct results are
# not hidden. The tradeoff is deliberate: on a public demo, hiding a good
# result is worse than showing a weak one, and signal 1 plus the displayed
# per-word confidences still cover genuinely unfamiliar input.
# UNVERIFIED: OOD recall at this threshold has not been measured for the
# phone model -- the 3 non-GRID clips behind the old figure were not
# re-scored. Treat as provisional ([[grid-demo-ood-calibration]]).
OOD_STD_THRESHOLD = 0.20

# Post-hoc word-confidence recalibration (1best only): the raw score from
# _word_posteriors/_word_confidences_1best is well-calibrated at the very
# top of its range but overconfident by 9-22 points through 0.4-0.9 (see
# [[confidence-calibration]] -- AUC-tuned, never checked for calibration
# until now). Isotonic regression fit on GRID test clips, held out at the
# clip level; cut ECE 0.041 -> 0.024 on clips never used to fit it.
# conformer_ctc2/_confidence_recalibrate.py regenerates this file.
#
# The map is fitted to one model's score distribution and is NOT portable
# across models: pairing a calibration with a different checkpoint silently
# reports wrong confidences. Override with --word-conf-calibration when
# serving anything other than the BPE-58 system this default was fitted on
# (the phone system uses word_conf_isotonic_calibration_phone_l8.npz).
WORD_CONF_CALIBRATION_PATH = (
    Path(__file__).resolve().parent / "word_conf_isotonic_calibration.npz"
)


def load_tokens(path: Path):
    """Parse tokens.txt into (id -> symbol dict, num_classes).

    Lexicon disambiguation symbols ('#0', '#1', ...) are skipped: they belong to
    the L/HLG graph, not the model's CTC output, so num_classes must not count
    them (e.g. GRID bpe-58 has 58 output classes but 60 lines in tokens.txt).
    """
    id2sym = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            sym, idx = parts[0], int(parts[1])
            if sym.startswith("#"):  # lexicon disambig symbol, not a model class
                continue
            id2sym[idx] = sym
    if not id2sym:
        raise ValueError(f"No tokens parsed from {path}")
    return id2sym, max(id2sym) + 1


def ctc_greedy_decode(nnet_output: torch.Tensor, id2sym: dict):
    """(1, T, C) log-probs -> (text, word_confs): argmax, collapse repeats,
    drop blanks. word_confs is a list of (word, confidence) where confidence
    is the geometric mean of the word's token posteriors (max over each
    token's frame run).

    Mirrors pretrained.py's token->word mapping (BPE '▁' -> space).
    """
    log_probs, ids = nnet_output[0].max(dim=-1)
    tokens = []  # [symbol, best frame log-prob over the token's run]
    prev = None
    for t, i in enumerate(ids.tolist()):
        if i != BLANK_ID:
            if i != prev:
                tokens.append([id2sym.get(i, ""), log_probs[t].item()])
            else:
                tokens[-1][1] = max(tokens[-1][1], log_probs[t].item())
        prev = i
    # Group BPE tokens ('▁' marks a word start) into words.
    words = []  # [word, [token log-probs]]
    for sym, lp in tokens:
        if sym.startswith("▁") or not words:
            words.append([sym.replace("▁", ""), [lp]])
        else:
            words[-1][0] += sym
            words[-1][1].append(lp)
    word_confs = [
        (w, math.exp(sum(lps) / len(lps))) for w, lps in words if w
    ]
    return " ".join(w for w, _ in word_confs), word_confs


def roi_strip(roi: np.ndarray, max_frames: int = 16) -> np.ndarray:
    """Sample up to max_frames evenly and tile them into one (H, W*k) image."""
    T = roi.shape[0]
    idx = np.linspace(0, T - 1, min(T, max_frames)).round().astype(int)
    return np.concatenate([roi[i] for i in idx], axis=1)


class LipReader:
    """Loads the AV-HuBERT front-end + Conformer-CTC model once, reusable per clip."""

    def __init__(self, args: argparse.Namespace):
        g = load_globals_avbubert(args)  # {device, model (AV-HuBERT, eval), transform}
        g.update(load_globals_dlib(args))  # {detector, predictor}
        self.detector = g["detector"]
        self.predictor = g["predictor"]
        # Robust per-frame fallback, tried only when the fast HOG detector
        # fails (glasses, tight framing, mild pose). Missing/broken MediaPipe
        # degrades to the old behaviour (no fallback), never to a crash.
        self.fallback_landmarker = None
        if getattr(args, "mediapipe_fallback", True):
            try:
                self.fallback_landmarker = _load_mediapipe_fallback()
                logging.info("MediaPipe fallback landmarker enabled")
            except Exception:
                logging.exception(
                    "MediaPipe fallback unavailable; continuing without"
                )
        self.avhubert = g["model"]
        self.transform = g["transform"]
        self.device = g["device"]
        self.layer = args.layer
        self.normalize_head = args.normalize_head
        # Mouth-ROI geometry. "meanface" aligns every frame to a reference face
        # before cropping (see local/meanface_roi.py) and must match whatever
        # the loaded checkpoint was trained on -- a model trained on one
        # geometry decodes the other poorly.
        # Feature mean normalisation; must match the checkpoint's training.
        self.cmn = getattr(args, "cmn", "none")
        self.roi_mode = getattr(args, "roi_mode", "centroid")
        self.mean_face = None
        if self.roi_mode == "meanface":
            from meanface_roi import load_mean_face  # local/ is on sys.path
            self.mean_face = load_mean_face(args.mean_face)
            logging.info(f"Mouth ROI: mean-face aligned (from {args.mean_face})")

        self.id2sym, num_classes = load_tokens(args.tokens)

        model = Conformer(
            num_features=768,
            num_classes=num_classes,
            subsampling_factor=1,
            d_model=args.encoder_dim,
            nhead=8,
            dim_feedforward=1024,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
        )
        load_checkpoint(str(args.checkpoint), model)
        self.model = model.to(self.device).eval()
        logging.info(
            f"Loaded model from {args.checkpoint} "
            f"(num_classes={num_classes}, layer={self.layer})"
        )

        # Optional word-confidence recalibration (see
        # WORD_CONF_CALIBRATION_PATH above). Missing/broken file degrades
        # to raw (uncalibrated) scores, never a crash.
        self._word_conf_calib = None
        calib_path = getattr(args, "word_conf_calibration", None) \
            or WORD_CONF_CALIBRATION_PATH
        try:
            data = np.load(calib_path)
            self._word_conf_calib = (data["x_sorted"], data["fitted_y"])
            logging.info(
                f"Loaded word-confidence calibration from {calib_path}"
            )
        except Exception:
            logging.exception(
                "Word-confidence calibration unavailable; using raw scores"
            )

        # Optional lexicon-constrained decoding graph (HLG).
        self.method = args.method
        self.HLG = None
        self.word_table = None
        if self.method == "1best":
            if args.HLG is None or args.words_file is None:
                raise ValueError("--method 1best requires --HLG and --words-file")
            import k2

            self.HLG = k2.Fsa.from_dict(
                torch.load(args.HLG, map_location="cpu", weights_only=False)
            ).to(self.device)
            if not hasattr(self.HLG, "lm_scores"):
                self.HLG.lm_scores = self.HLG.scores.clone()
            self.word_table = k2.SymbolTable.from_file(str(args.words_file))
            logging.info(
                f"Loaded HLG from {args.HLG} and words from {args.words_file}"
            )

        # Idle GPU offload: on a rarely-used public demo, keeping the models
        # resident wastes GPU memory that other jobs on this box could use.
        # After `idle_offload_seconds` with no request, move the three
        # GPU-resident objects (AV-HuBERT, Conformer, HLG) to CPU and free the
        # CUDA cache; the next request moves them back (<1 s). A single lock
        # serialises this against inference (gradio already runs one GPU job at
        # a time), so a request can never see a half-migrated model. Disabled
        # (`<= 0`) or when there is no CUDA device, this is a no-op.
        self._gpu_device = self.device
        self._cpu_device = torch.device("cpu")
        self._device_lock = threading.RLock()
        self._offloaded = False
        self._last_active = time.monotonic()
        self._offload_seconds = getattr(args, "idle_offload_seconds", 0)
        if self._offload_seconds > 0 and self._gpu_device.type == "cuda":
            t = threading.Thread(target=self._idle_offload_loop, daemon=True)
            t.start()
            logging.info(
                f"Idle GPU offload enabled: models move to CPU after "
                f"{self._offload_seconds}s idle"
            )

    def _move_models(self, device: torch.device) -> None:
        """Move the GPU-resident objects to `device`. Caller holds the lock."""
        self.avhubert.to(device)
        self.model.to(device)
        if self.HLG is not None:
            self.HLG = self.HLG.to(device)
        self.device = device

    def _activate(self) -> None:
        """Ensure the models are on the GPU before an inference. Caller holds
        the lock. Also stamps activity so the idle timer restarts."""
        if self._offloaded:
            logging.info("Request arrived; restoring models to GPU")
            self._move_models(self._gpu_device)
            self._offloaded = False
        self._last_active = time.monotonic()

    def _idle_offload_loop(self) -> None:
        """Background daemon: offload the models to CPU once the demo has been
        idle for `_offload_seconds`. Re-checks under the lock so it never
        offloads a model that a request just reactivated."""
        while True:
            time.sleep(min(30, self._offload_seconds))
            with self._device_lock:
                if self._offloaded:
                    continue
                idle = time.monotonic() - self._last_active
                if idle >= self._offload_seconds:
                    logging.info(
                        f"Idle for {idle:.0f}s; offloading models to CPU"
                    )
                    self._move_models(self._cpu_device)
                    self._offloaded = True
                    torch.cuda.empty_cache()

    def _roi_and_features(self, video: str, noise_sigma: float = 0.0,
                          blur_sigma: float = 0.0):
        """Return (raw ROI frames (T,88,88) uint8, AV-HuBERT features (T,768))."""
        # Temporal normalization: the model was trained on 25 fps GRID video,
        # so resample other frame rates (e.g. 30/60 fps uploads) before
        # landmarking. No-op for GRID clips.
        resampled = None
        fps = _video_fps(video)
        if fps and abs(fps - TARGET_FPS) > 0.1:
            resampled = _resample_to_target_fps(video)
            if resampled is None:
                logging.warning(
                    f"{video} is {fps:.2f} fps and could not be resampled to "
                    f"{TARGET_FPS:g} fps (ffmpeg missing or failed); "
                    "recognition accuracy may degrade."
                )
            else:
                logging.info(f"Resampled {video}: {fps:.2f} -> {TARGET_FPS:g} fps")
        try:
            return self._roi_and_features_25fps(
                video, Path(resampled) if resampled else Path(video),
                noise_sigma, blur_sigma,
            )
        finally:
            if resampled is not None:
                Path(resampled).unlink(missing_ok=True)

    def _roi_and_features_25fps(self, video: str, video_path: Path,
                                noise_sigma: float = 0.0,
                                blur_sigma: float = 0.0):
        """`_roi_and_features` body; video_path is the (possibly resampled)
        clip to process, video the original name for messages."""
        landmarks = _detect_landmarks(
            video_path, self.detector, self.predictor, 1,
            detect_height=DETECT_HEIGHT,
            fallback_landmarker=self.fallback_landmarker,
        )
        if landmarks is None:
            logging.warning(f"No face detected in {video}")
            raise RuntimeError(
                "no face was detected in the clip. Please use a frontal "
                "talking-face video with the whole face visible."
            )
        if self.roi_mode == "meanface":
            # Pose normalization, not just scale: every frame is warped onto a
            # reference face (similarity fit on eye corners + nose tip) before
            # the mouth crop, which is what AV-HuBERT itself was pretrained on.
            from meanface_roi import extract_meanface_frames
            roi = extract_meanface_frames(video_path, landmarks, self.mean_face)
            if roi.shape[0] == 0:
                raise RuntimeError(
                    "the mouth region could not be aligned in this clip. "
                    "Please use a frontal talking-face video."
                )
            return self._features_from_roi(roi, noise_sigma, blur_sigma)
        # Head-size normalization: scale the mouth crop to the detected face
        # size (via IOD) so the mouth-to-head ratio matches GRID regardless of
        # how big the face is in the frame. On GRID this yields ~64px (no-op).
        if self.normalize_head:
            crop = max(16, round(CROP_PER_IOD * _median_iod(landmarks)))
            logging.info(f"Head-normalized mouth crop: {crop}px "
                         f"(IOD {_median_iod(landmarks):.1f}px)")
        else:
            crop = MOUTH_W
        raw = _extract_mouth_frames(
            video_path, landmarks, crop, crop, ROI_SIZE, MOUTH_LEFT, MOUTH_RIGHT
        )
        return self._features_from_roi(np.stack(raw), noise_sigma, blur_sigma)

    def _features_from_roi(self, roi: np.ndarray, noise_sigma: float = 0.0,
                           blur_sigma: float = 0.0):
        """Shared tail of both ROI geometries: optional degradation, then
        AV-HuBERT. Returns (T,88,88) uint8 ROI and (T,768) features."""
        if noise_sigma or blur_sigma:  # robustness playground
            roi = _degrade_roi(roi, noise_sigma, blur_sigma)
        frames = self.transform(roi)  # normalized float (T, 88, 88)
        tensor = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats, _ = self.avhubert.extract_finetune(
                source={"video": tensor, "audio": None},
                padding_mask=None,
                output_layer=self.layer,
            )
        return roi, feats.squeeze(0)  # (T,88,88), (T,768)

    def _decode_1best(self, nnet_output: torch.Tensor):
        """Lexicon-constrained decoding via HLG + one-best path (like decode.py).

        Returns (text, word_confs)."""
        from icefall.decode import one_best_decoding
        from icefall.utils import get_texts

        lattice = self._build_lattice(nnet_output, OUTPUT_BEAM)
        best_path = one_best_decoding(lattice=lattice, use_double_scores=True)
        hyp = get_texts(best_path)[0]
        words = [self.word_table[i] for i in hyp]
        try:
            # Confidence lattice: see CONF_OUTPUT_BEAM/CONF_TEMPERATURE.
            conf_nnet = torch.log_softmax(
                nnet_output / CONF_TEMPERATURE, dim=-1
            )
            conf_lattice = self._build_lattice(conf_nnet, CONF_OUTPUT_BEAM)
            word_confs = self._word_posteriors(conf_lattice, hyp, words)
        except Exception:
            logging.exception("Lattice word posteriors failed; falling back")
            word_confs = self._word_confidences_1best(
                best_path, nnet_output, words
            )
        word_confs = self._calibrate_word_confs(word_confs)
        try:
            _, emit_nb = self._emission_score(best_path, nnet_output)
            low_emission = emit_nb < OOD_HIDE_THRESHOLD
        except Exception:
            logging.exception("OOD emission score failed")
            low_emission = False
        try:
            nb_std = self._confidence_stability(nnet_output)
            unstable_confidence = nb_std > OOD_STD_THRESHOLD
        except Exception:
            logging.exception("OOD confidence-stability score failed")
            unstable_confidence = False
        ood = "hide" if (low_emission or unstable_confidence) else ""
        return " ".join(words), word_confs, ood

    def _calibrate_word_confs(self, word_confs):
        """Map raw word-confidence scores through the fitted isotonic
        calibration (WORD_CONF_CALIBRATION_PATH), if one loaded; a no-op
        otherwise. Only meaningful for scores from _word_posteriors /
        _word_confidences_1best -- the calibration was fit on those, not
        on ctc_greedy_decode's differently-computed scores."""
        if self._word_conf_calib is None:
            return word_confs
        x_sorted, fitted_y = self._word_conf_calib
        calibrated = []
        for w, c in word_confs:
            idx = np.searchsorted(x_sorted, c, side="right") - 1
            idx = int(np.clip(idx, 0, len(fitted_y) - 1))
            calibrated.append((w, float(fitted_y[idx])))
        return calibrated

    def _emission_score(self, best_path, nnet_output):
        """Out-of-domain signal: how well the observed lip frames match the
        decoded GRID sentence, independent of the grammar. The best path is
        frame-synchronous, so each arc's token label aligns with a frame;
        we average the visual model's log-prob of that token over frames.
        Returns (mean over all frames, mean over non-blank frames only);
        both are audio-free (nnet_output is the visual model output). Lower
        (more negative) means the frames are poorly explained -> likely not
        a GRID utterance."""
        fsa = best_path[0]
        labels = [l for l in fsa.labels.tolist() if l != -1]  # token/frame
        lp = nnet_output[0].cpu()  # (T, C) log-probs
        T = min(len(labels), lp.shape[0])
        allf = [lp[t, labels[t]].item() for t in range(T)]
        nonblank = [lp[t, labels[t]].item()
                    for t in range(T) if labels[t] != BLANK_ID]
        mean_all = sum(allf) / len(allf) if allf else float("-inf")
        mean_nb = (sum(nonblank) / len(nonblank) if nonblank
                   else float("-inf"))
        return mean_all, mean_nb

    def _confidence_stability(self, nnet_output: torch.Tensor) -> float:
        """Second, independent out-of-domain signal: std of the per-frame
        max-softmax (top-1 probability at each frame) over non-blank
        frames. Unlike `_emission_score`, this reads nnet_output directly
        and doesn't depend on which token the HLG grammar forced onto each
        frame. Genuine GRID speech keeps this low even when the decoded
        word is wrong (stable confidence, wrong content); unfamiliar mouth
        motion makes per-frame confidence swing, raising it. Higher ->
        more likely out-of-domain."""
        lp = nnet_output[0].cpu()  # (T, C) log-probs
        max_lp, ids = lp.max(dim=-1)
        nonblank = ids != BLANK_ID
        p_max_nb = max_lp[nonblank].exp()
        if p_max_nb.numel() < 2:
            return 0.0
        return p_max_nb.std().item()

    def _build_lattice(self, nnet_output: torch.Tensor, output_beam: int):
        from icefall.decode import get_lattice

        T = nnet_output.shape[1]
        # Single, unpadded clip: one segment covering all T frames.
        supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
        return get_lattice(
            nnet_output=nnet_output,
            decoding_graph=self.HLG,
            supervision_segments=supervision_segments,
            search_beam=SEARCH_BEAM,
            output_beam=output_beam,
            min_active_states=MIN_ACTIVE_STATES,
            max_active_states=MAX_ACTIVE_STATES,
            subsampling_factor=1,
        )

    def _word_posteriors(self, lattice, hyp_ids, words):
        """Marginal word posteriors from the full decoding lattice.

        Forward-backward (via k2 arc posteriors in the log semiring) gives
        each arc's posterior mass; a word's confidence is the total mass of
        all lattice arcs emitting it. Unlike a best-path score, this accounts
        for competing hypotheses: if 'n' and 'l' split the letter slot, both
        score ~0.5 instead of the winner claiming ~1.0. Relies on each word
        occurring at most once per path, which GRID's disjoint slot
        vocabularies guarantee.
        """
        import k2

        # (num_arcs,) log posteriors under the full lattice.
        arc_post = lattice.get_arc_post(
            use_double_scores=True, log_semiring=True
        )
        post = arc_post.exp()
        aux = lattice.aux_labels
        if isinstance(aux, k2.RaggedTensor):
            vals = aux.values.long()
            contrib = post[aux.shape.row_ids(1).long()]
        else:
            vals = aux.long()
            contrib = post
        word_confs = []
        for wid, w in zip(hyp_ids, words):
            p = contrib[vals == wid].sum().item()
            word_confs.append((w, min(1.0, p)))
        return word_confs

    def _word_confidences_1best(self, best_path, nnet_output, words):
        """Per-word confidence from the one-best path.

        The lattice is frame-synchronous (each non-final arc consumes one
        frame), so arc i's token label aligns with frame i. A word spans from
        the frame where its id is emitted (aux_label > 0) to the next word's
        start; its confidence is the geometric mean of the CTC posteriors of
        the span's non-blank tokens. Returns [] if the arc/frame alignment
        doesn't hold, in which case confidences are simply omitted.
        """
        import k2

        fsa = best_path[0]
        labels = fsa.labels.tolist()  # token per frame; final arc is -1
        aux = fsa.aux_labels
        if isinstance(aux, k2.RaggedTensor):
            aux_rows = aux.tolist()
        else:
            aux_rows = [[a] for a in aux.tolist()]

        T = nnet_output.shape[1]
        frame_labels = [l for l in labels if l != -1]
        if len(frame_labels) != T:
            return []
        starts = []
        for frame, (lab, auxs) in enumerate(zip(labels, aux_rows)):
            if lab == -1:
                continue
            starts.extend(frame for a in auxs if a > 0)
        if len(starts) != len(words):
            return []

        lp = nnet_output[0].cpu()  # (T, C) log-probs
        bounds = starts + [T]
        word_confs = []
        for w, s, e in zip(words, bounds, bounds[1:]):
            lps = [
                lp[f, frame_labels[f]].item()
                for f in range(s, e)
                if frame_labels[f] != BLANK_ID
            ]
            if not lps:  # span is all blank; use its frames as-is
                lps = [lp[f, frame_labels[f]].item() for f in range(s, e)]
            word_confs.append((w, math.exp(sum(lps) / len(lps))))
        return word_confs

    @torch.no_grad()
    def recognize(self, video: str, noise_sigma: float = 0.0,
                  blur_sigma: float = 0.0):
        """Return (recognised_text, ROI frames as recognized, word_confs, ood).

        word_confs is a list of (word, confidence in [0, 1]) pairs; it may be
        empty when confidences could not be derived. ood is '' (in-domain),
        'warn' (looks out-of-domain, still show the closest guess) or 'hide'
        (definitely not a GRID sentence); None for greedy decoding, which has
        no lattice to score."""
        # Backstop for the CLI/API path (the UI enforces source-specific
        # limits up front); use the more lenient bound as a pure safety net.
        err = _duration_error(video, MAX_DURATION_WEBCAM_S)
        if err:
            raise ValueError(err)
        # Hold the device lock across the whole inference so the idle monitor
        # cannot offload the models mid-request; `_activate` restores them to
        # the GPU first if they were offloaded, and refreshes the idle timer.
        with self._device_lock:
            self._activate()
            try:
                roi, feats = self._roi_and_features(video, noise_sigma, blur_sigma)
                feature = feats.unsqueeze(0).to(self.device)  # (1, T, 768)
                # Must match the --cmn the checkpoint was trained with.
                feature = apply_cmn(feature, self.cmn)
                nnet_output = self.model(feature, None)[0]  # (1, T, C)
                if self.method == "1best":
                    text, word_confs, ood = self._decode_1best(nnet_output)
                else:
                    text, word_confs = ctc_greedy_decode(
                        nnet_output.cpu(), self.id2sym
                    )
                    ood = None
            finally:
                # Restart the idle countdown from when the request finished,
                # not when it started (long clips shouldn't shorten it).
                self._last_active = time.monotonic()
        return text, roi, word_confs, ood


def _video_fps(video: str) -> float:
    """Container frame rate as reported by OpenCV (0.0 if unknown)."""
    import cv2

    cap = cv2.VideoCapture(video)
    try:
        return cap.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        cap.release()


def _video_duration(video: str) -> float:
    """Duration in seconds. Uses container metadata (fast, works for normal
    files); falls back to reading packet timestamps with ffprobe for
    variable-frame-rate clips like browser webcam WebM, whose containers
    store no usable duration (OpenCV returns a garbage frame count).
    Returns 0.0 only if both fail."""
    import cv2

    cap = cv2.VideoCapture(video)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps > 0 and 0 < frames < 1e7:
            return frames / fps
    finally:
        cap.release()
    return _ffprobe_duration(video)


def _ffprobe_duration(video: str) -> float:
    """Duration from the last video packet's timestamp (demux only, no pixel
    decode). Robust for VFR/WebM. 0.0 if ffprobe is missing or fails."""
    import shutil
    import subprocess

    if shutil.which("ffprobe") is None:
        return 0.0
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", video],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    times = [float(t) for t in out.split() if t and t != "N/A"]
    return max(times) if times else 0.0


def _duration_error(video: str, max_duration: float = MAX_DURATION_S):
    """Rejection message if the clip's length is outside the accepted range,
    None if acceptable (or the duration is unknown)."""
    duration = _video_duration(video)
    if duration and not (
        MIN_DURATION_S - DURATION_TOL_S
        <= duration
        <= max_duration + DURATION_TOL_S
    ):
        return (
            f"Clip is {duration:.1f}s long; please use a clip between "
            f"{MIN_DURATION_S:g}s and {max_duration:g}s."
        )
    return None


def _resample_to_target_fps(src: str):
    """Re-encode src at TARGET_FPS (video only). Returns the path to a temp
    file the caller must delete, or None if ffmpeg is unavailable/fails."""
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        return None
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    try:
        # mpeg4 (not libx264): handles odd frame dimensions, and the output is
        # only read back by OpenCV, never a browser.
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-vf", f"fps={TARGET_FPS:g}",
             "-an", "-c:v", "mpeg4", "-q:v", "2", out],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError):
        Path(out).unlink(missing_ok=True)
        return None
    return out


# Per-request temp dirs for UI output files. Gradio 4 copies returned files
# into its own cache before serving, so dirs from older requests can be
# deleted; keeping the last few bounds disk use on a long-running server.
_REQUEST_DIRS: list = []
_MAX_REQUEST_DIRS = 4


def _new_request_dir() -> str:
    import shutil
    import tempfile

    while len(_REQUEST_DIRS) >= _MAX_REQUEST_DIRS:
        shutil.rmtree(_REQUEST_DIRS.pop(0), ignore_errors=True)
    d = tempfile.mkdtemp(prefix="vsr-demo-")
    _REQUEST_DIRS.append(d)
    return d


def _log_activity(path, source: str, word_confs, ood) -> None:
    """Append one anonymous line per submission: timestamp, source
    (upload/webcam), and the per-word confidence scores as bare numbers.
    Deliberately content-free: no IP, no identity, and no words/transcript
    (only the numeric confidences), so it keeps no recording of what was
    said. 'ood' when the clip was flagged out-of-domain, '-' when there are
    no scores."""
    from datetime import datetime

    if path is None:
        return
    if ood == "hide":
        confs = "ood"
    elif word_confs:
        confs = ",".join(f"{c:.2f}" for _, c in word_confs)
    else:
        confs = "-"
    line = "\t".join(
        [datetime.now().isoformat(timespec="seconds"), source, confs]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _to_playable_mp4(src: str, out_dir=None):
    """Transcode any clip to H.264/yuv420p mp4 so browsers can play it back.

    GRID clips are .mpg (MPEG-1), which OpenCV reads but HTML5 <video> cannot
    play; re-encode to H.264 for the UI. Returns None if ffmpeg is unavailable.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        return None
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=out_dir).name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-movflags", "+faststart", out],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError):
        Path(out).unlink(missing_ok=True)
        return None
    return out


def _frames_to_mp4(frames: np.ndarray, fps: int = 25, scale: int = 3, out_dir=None):
    """Encode grayscale ROI frames (T,H,W uint8) to an H.264 mp4 for the UI.

    Uses ffmpeg (raw frames via stdin), nearest-neighbour upscaled by `scale`
    so the 88x88 ROI is viewable, with +faststart for in-browser playback.
    Returns None if ffmpeg is unavailable or encoding fails.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        return None
    T, H, W = frames.shape
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=out_dir).name
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0",
        "-vf", f"scale=iw*{scale}:ih*{scale}:flags=neighbor",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out,
    ]
    try:
        p = subprocess.run(
            cmd, input=np.ascontiguousarray(frames, dtype=np.uint8).tobytes(),
            capture_output=True,
        )
    except OSError:
        Path(out).unlink(missing_ok=True)
        return None
    if p.returncode != 0:
        Path(out).unlink(missing_ok=True)
        return None
    return out


def launch_ui(reader: LipReader, example_paths=None,
              hour_limit=10, day_limit=30, activity_log=None):
    import gradio as gr

    def infer(video, source, noise, blur, request: gr.Request):
        # Last three slots: Submit interactivity, the input video itself
        # (cleared when the clip was the problem, kept otherwise), and the
        # inline status message shown under the video (empty = nothing to
        # say). Messages are typed into status_msg, not popped as a corner
        # toast, so they read as part of the video area.
        blank = ("", None, None, None)
        if not video:
            return (*blank, gr.update(interactive=False), gr.update(),
                    gr.update())
        ip = _client_ip(request)
        local = _is_local_ip(ip)
        window = None if local else _quota_exceeded(ip, hour_limit, day_limit)
        if window == "hourly":
            # Nothing wrong with the clip; keep it loaded.
            return (*blank, gr.update(interactive=False), gr.update(),
                    f"Hourly limit reached ({hour_limit} videos per visitor "
                    "per hour). Please try again in a little while.")
        if window == "daily":
            return (*blank, gr.update(interactive=False), gr.update(),
                    f"Daily limit reached ({day_limit} videos per visitor "
                    "per day). Please try again tomorrow.")
        try:
            result = _infer(video, source, noise, blur)
            return result
        except gr.Error as e:
            # Explain inline WITHOUT raising: raising stamps a red "Error"
            # badge on every output component. The faulty clip is cleared,
            # returning the upload area to its normal empty state (the
            # change handler then resets Submit).
            return (*blank, gr.update(), None, e.message)
        except Exception:
            # Never let a bare, unexplained "Error" reach the user.
            logging.exception(f"Unexpected failure processing {video}")
            return (*blank, gr.update(), None,
                    "Processing failed unexpectedly; the problem has been "
                    "logged. Please try again with a different clip.")

    def _infer(video, source, noise=0.0, blur=0.0):
        # Nothing of the user's is stored: the clip is recognized in memory
        # and the only files written are transient outputs served back to the
        # browser (auto-purged), plus an anonymous, content-free activity log
        # (timestamp, upload/webcam, confidence numbers only).
        try:
            text, roi, word_confs, ood = reader.recognize(video, noise, blur)
        except Exception as e:
            logging.exception(f"Recognition failed for {video}")
            raise gr.Error(f"Recognition failed: {e}")
        try:
            _log_activity(activity_log, source, word_confs, ood)
        except OSError:
            logging.exception("Activity log write failed")  # non-fatal
        if ood == "hide":
            # Definitely not a GRID sentence: show nothing rather than a
            # misleading confident guess. Clip stays loaded, Submit off.
            return ("", None, None, None,
                    gr.update(interactive=False), gr.update(),
                    "This doesn't look like a GRID sentence, so there's "
                    "nothing to show. Please speak a sentence in the fixed "
                    "command·colour·preposition·letter·digit·adverb "
                    "pattern.")
        out_dir = _new_request_dir()
        return (
            text.title(),  # display: Title Case
            # Score in the text (gradio only tints, never prints, labels) plus
            # a bucket category for the colour: float labels tint red like a
            # saliency map, so use color_map'd buckets instead.
            [
                (f"{w.title()} {c:.2f}",
                 "high" if c >= 0.9 else "medium" if c >= 0.7 else "low")
                for w, c in word_confs
            ] or None,
            _frames_to_mp4(roi, out_dir=out_dir),
            roi_strip(roi),
            # This clip is done; Submit stays off until a new one arrives
            # (re-enabled by video_in.change below). The clip stays loaded.
            gr.update(interactive=False),
            gr.update(),
            "",  # clear any stale status message from a previous attempt
        )

    # Example clips: GRID .mpg won't play in the browser, so show a transcoded
    # mp4 copy (kept for the server's lifetime) when ffmpeg is available.
    examples = []
    for pth in example_paths or []:
        if not Path(pth).exists():
            logging.warning(f"Example clip not found, skipping: {pth}")
            continue
        examples.append([_to_playable_mp4(str(pth)) or str(pth)])

    # Only the input component has a .source-selection bar (upload/webcam
    # icons); keep it un-clipped and its icons comfortably visible.
    css = """
    .source-selection { height: var(--size-12) !important; flex-shrink: 0; }
    .source-selection .icon { width: 30px !important; height: 30px !important; }
    /* Letterbox the live webcam preview instead of crop-zooming it to fill
       the container, which distorts/flattens the view. */
    video { object-fit: contain !important; }
    /* Never break a word-confidence chip across lines; wrap whole chips. */
    .word-conf span { display: inline-block; white-space: nowrap; }
    /* Prediction textbox: large and bold, it is the main output. */
    .rec-text textarea { font-weight: 700; font-size: 1.5em; }
    /* Bold every component title/label (Prediction, Word confidence, Mouth
       ROI, sliders, ...) via gradio's own weight vars, not hashed classes. */
    .gradio-container { --block-title-text-weight: 700;
        --block-label-text-weight: 700; }
    /* Privacy note: accent-tinted so it reads as a reassurance, not fine print. */
    .privacy-note p { color: var(--color-accent); font-weight: 600;
        margin: 4px 0; }
    /* Inline clip status/rejection message, overlaid as a banner on top of
       the video area instead of a corner toast; empty when there is
       nothing to say. #video-wrap is the positioning root (see the
       gr.Group around video_in/status_msg); top, not bottom, so it never
       covers the source-select / native video-player controls, which are
       always at the bottom. */
    #video-wrap { position: relative; }
    /* top: 36px, not 0, so the banner clears the component's own header row
       -- the label and, at its right, the clear (X) that returns the upload
       area to the chooser. At 0 the banner covered the X, hiding the one
       control that recovers from a rejected clip. */
    .status-msg { position: absolute; left: 0; right: 0; top: 36px;
        z-index: 20; pointer-events: none; }
    .status-msg p { margin: 0; padding: 10px 14px;
        background: rgba(0, 0, 0, 0.72); color: #fff; font-weight: 600;
        text-align: center; border-radius: var(--radius-lg) var(--radius-lg) 0 0; }
    /* Hide the gradio footer (Built with Gradio / Use via API / Settings). */
    footer { display: none !important; }
    /* Hide the video trim (scissors) control on the input player. */
    button[aria-label="Trim video to selection"] { display: none !important; }
    /* Hide the persistent in-component "Error" status pill (client-side
       upload/processing failures); toasts still explain what went wrong.
       Scoped to component status overlays so error toasts are unaffected. */
    .wrap .error { display: none !important; }
    """
    # Client-side upload/processing failures leave a component stuck in an
    # error state the server cannot see or reset. Auto-click its clear (X)
    # control so the upload area snaps back to the normal drop zone; the
    # matching CSS above hides the transient "Error" pill.
    js = """
    () => {
      // Grammar hint shown over the upload area while it waits for a video.
      // Static explanation with only the title-cased example sentence
      // rotating through randomly generated grammar-valid sentences.
      const cmd = ['bin','lay','place','set'],
            col = ['blue','green','red','white'],
            prep = ['at','by','in','with'],
            letters = 'abcdefghijklmnopqrstuvxyz'.split(''),
            dig = ['zero','one','two','three','four','five','six','seven',
                   'eight','nine'],
            adv = ['again','now','please','soon'];
      const pick = a => a[Math.floor(Math.random() * a.length)];
      const cap = w => w.charAt(0).toUpperCase() + w.slice(1);
      const ticker = document.getElementById('sentence-ticker');
      if (ticker && !ticker.dataset.init) {
        ticker.dataset.init = '1';
        ticker.innerHTML =
          'A valid sentence uses a 6-word grammatical structure: '
          + 'command | colour | preposition | letter | number | adverb, '
          + 'such as \\u201C<span id="ticker-eg"></span>\\u201D';
        const eg = document.getElementById('ticker-eg');
        eg.style.transition = 'opacity 0.35s';
        const tick = () => {
          eg.style.opacity = 0;
          setTimeout(() => {
            eg.textContent = [pick(cmd), pick(col), pick(prep),
              pick(letters), pick(dig), pick(adv)].map(cap).join(' ');
            eg.style.opacity = 1;
          }, 350);
        };
        tick();
        setInterval(tick, 3500);
      }
      // Show the hint only while the input is empty (no <video> loaded).
      const toggleHint = () => {
        const vin = document.getElementById('video-input');
        if (vin && ticker) {
          ticker.style.display =
            vin.querySelector('video') ? 'none' : '';
        }
      };
      const resetErrored = () => {
        document.querySelectorAll('.block .error').forEach(err => {
          const block = err.closest('.block');
          if (!block) return;
          // The X is an icon button nested inside the .clear-status div;
          // clicking the div itself does nothing.
          const btn = block.querySelector('.clear-status button')
                   || block.querySelector('button.clear-status');
          if (btn) btn.click();
        });
      };
      const onMutate = () => { resetErrored(); toggleHint(); };
      onMutate();
      new MutationObserver(onMutate)
        .observe(document.body, {subtree: true, childList: true});
    }
    """
    # delete_cache purges gradio's upload/output cache (every 10 minutes,
    # files older than 15) -- gradio never cleans it on its own, and the
    # cache lives on a bounded tmpfs that upload bursts could otherwise fill.
    # Vivid indigo accent so the primary Submit button reads as active (and
    # its disabled state as muted-colour, not "broken"); recolours the ticker
    # and grammar-hint accents to match.
    theme = gr.themes.Default(primary_hue="indigo")
    with gr.Blocks(
        title="Can AI Read Your Lips? A Live Lipreading Demo "
              "(based on GRID corpus)",
        delete_cache=(600, 900), css=css, js=js, theme=theme,
    ) as demo:
        gr.Markdown("# Can AI Read Your Lips?")
        gr.Markdown(
            "**Live visual speech recognition constrained to the GRID "
            "corpus grammar.**",
            elem_classes=["subtitle"],
        )
        with gr.Row():
            with gr.Column():
                # elem_id="video-wrap" is the CSS positioning root for
                # status_msg below, which overlays this group as a banner.
                with gr.Group(elem_id="video-wrap"):
                    video_in = gr.Video(
                        # Viewport-relative cap so the player plus the
                        # Submit button below it stay above the fold on a
                        # phone; 410px on any window taller than ~910px.
                        label="Frontal talking-face clip",
                        height="min(410px, 45vh)",
                        elem_id="video-input",
                        sources=["webcam", "upload"],
                    )
                    # Rejection/status messages for this clip (bad duration,
                    # unreadable file, quota, processing failure, OOD).
                    # Overlaid on the video area itself (top banner) rather
                    # than a corner toast or a line of text below it.
                    status_msg = gr.Markdown("", elem_classes=["status-msg"])
                # Directly under the upload/record controls so it is visible
                # without scrolling on a phone, where the column stacks and
                # everything below (examples, sliders) runs off-screen.
                with gr.Row():
                    # Enabled only while an unsubmitted clip is loaded: off at
                    # start, on when the input changes, off again after infer.
                    submit = gr.Button("Submit", variant="primary",
                                       interactive=False)
                    # Every finished submission -- result, out-of-domain, or
                    # quota -- deliberately keeps the clip loaded with Submit
                    # disabled, so without this the only way back to the
                    # upload/record chooser is the video component's small
                    # clear (X). Stays enabled after Submit goes dead; that
                    # is the whole point of it.
                    new_clip = gr.Button("Try another clip",
                                         interactive=False)
                # Grammar hint shown right under the upload area while it is
                # empty (js populates it and hides it once a clip loads).
                gr.HTML(
                    '<div id="sentence-ticker" style="text-align:center;'
                    'font-weight:600;padding:6px 4px;min-height:1.4em;'
                    'color:var(--color-accent)"></div>'
                )
                gr.Markdown(
                    "🔒 *Your video is processed in memory and never "
                    "stored. No recording, image, or transcript of what you "
                    "submit is kept.*",
                    elem_classes=["privacy-note"],
                )
                gr.Markdown(
                    "*Best with a single frontal face in good lighting "
                    f"({MIN_DURATION_S:g}–{MAX_DURATION_WEBCAM_S:g} s).*"
                )
                # "upload"/"webcam"; set by the upload/record events, used
                # for the max-duration cap and the anonymous activity log.
                source_state = gr.State("upload")
                if examples:
                    gr.Examples(
                        examples=examples, inputs=[video_in],
                        label="Example clips from speakers unseen in "
                              "training: s1, s2, s20, s22 & one from "
                              "Lombard Grid corpus & a live recording "
                              "(click & submit)",
                    )
                with gr.Accordion(
                    "Challenge the model: add noise or blur (optional)",
                    open=False,
                ):
                    gr.Markdown(
                        "These sliders worsen the video quality of the "
                        "mouth region before the model reads it, so you can "
                        "see how much degradation it tolerates. The Mouth "
                        "ROI panels show exactly what the model received. "
                        "Leave at 0 for normal recognition."
                    )
                    noise_sl = gr.Slider(0, 50, value=0, step=1,
                                         label="Noise (0 = none)")
                    blur_sl = gr.Slider(0, 5, value=0, step=0.25,
                                        label="Blur (0 = none)")
            # elem_id: the results are scrolled into view after a successful
            # recognition (see submit.click below), which matters on a phone
            # where this column stacks below the whole input column.
            with gr.Column(elem_id="results-col"):
                with gr.Row():
                    roi_vid = gr.Video(
                        label="Mouth ROI (animated)",
                        show_download_button=False, height=240, scale=1,
                    )
                    with gr.Column(scale=2):
                        text_out = gr.Textbox(label="Prediction",
                                              elem_classes=["rec-text"])
                        conf_out = gr.HighlightedText(
                            label="Word confidence",
                            color_map={"high": "green", "medium": "yellow",
                                       "low": "red"},
                            show_inline_category=False,
                            show_legend=False,
                            elem_classes=["word-conf"],
                        )
                strip_out = gr.Image(
                    label="Mouth ROI (sampled frames)", image_mode="L",
                    height=140, show_download_button=False,
                )
                # Grammar: the FST drawing is always visible, filling the
                # space in the input column; word tables sit behind the
                # accordion.
                gr.Markdown(
                    "### Valid GRID sentence structure\n"
                    "The model only understands 6-word sentences that follow "
                    "the pattern below, one word from each column, e.g. "
                    "**“place green with h eight now”** or "
                    "**“bin blue at f two now”**."
                )
                gr.HTML(_grammar_fst_svg())
                with gr.Accordion("Full grammar description", open=False):
                    gr.Markdown(GRID_GRAMMAR_MD)
        gr.Markdown(ACKNOWLEDGEMENTS_MD)

        def check_length(video, source):
            """Reject out-of-range clips as soon as they are supplied (by
            upload or by webcam recording), before Submit: clear the input
            and explain why inline (status_msg, overlaid on the video area),
            not as a corner toast. recognize() re-checks as a backstop for the CLI/API
            path. `source` selects the max-duration cap (webcam gets more)
            and is recorded, so it is echoed into source_state.

            On the accept path, video_in gets gr.update() (a true no-op),
            not the video it was passed: writing the same value back into
            the component re-triggers its own re-render (a visible repaint
            of the clip) and, since video_in.change() clears the other
            output panels, a redundant repaint of those too -- on top of
            the one .change() already does for the genuine new-clip event."""
            if video:
                max_dur = (MAX_DURATION_WEBCAM_S if source == "webcam"
                           else MAX_DURATION_S)
                try:
                    err = _duration_error(video, max_dur)
                except Exception:
                    logging.exception(f"Could not read clip {video}")
                    return (None, source, "This video could not be read; "
                            "please try a different file or recording.")
                if err:
                    return None, source, err
            return gr.update(), source, ""

        video_in.upload(
            lambda v: check_length(v, "upload"),
            inputs=[video_in], outputs=[video_in, source_state, status_msg],
        )
        video_in.stop_recording(
            lambda v: check_length(v, "webcam"),
            inputs=[video_in], outputs=[video_in, source_state, status_msg],
        )
        # Fires on any new value (upload, webcam recording, example click)
        # and on clearing: Submit is usable exactly when a clip is loaded,
        # and results from the previous clip are cleared.
        def on_clip_change(video):
            loaded = video is not None
            return (
                gr.update(interactive=loaded),
                gr.update(interactive=loaded),  # "Try another clip"
                "",    # recognised text
                None,  # word confidence
                None,  # ROI video
                None,  # ROI strip
            )

        video_in.change(
            on_clip_change, inputs=[video_in],
            outputs=[submit, new_clip, text_out, conf_out, roi_vid,
                     strip_out],
        )
        # Clearing the input brings back the upload/record chooser; the
        # change handler above then resets Submit and wipes the old results.
        # The status message is not one of its outputs, so clear it here.
        new_clip.click(lambda: (None, ""), outputs=[video_in, status_msg])
        # Changing the degradation makes re-submitting the same clip a new
        # experiment, so it re-enables Submit (if a clip is loaded).
        for slider in (noise_sl, blur_sl):
            slider.change(
                lambda v: gr.update(interactive=v is not None),
                inputs=[video_in], outputs=[submit],
            )
        submit.click(
            infer,
            inputs=[video_in, source_state, noise_sl, blur_sl],
            outputs=[text_out, conf_out, roi_vid, strip_out,
                     submit, video_in, status_msg],
        ).then(
            # Bring the result to the user instead of making them scroll and
            # guess: on a phone the results column sits below the entire
            # input column, so a finished prediction lands off-screen.
            # Client-side only (fn=None); gradio's own scroll_to_output is no
            # use here because it targets the topmost output component, which
            # is the video input itself. Skipped when there is no prediction
            # to show (rejected clip, quota, out-of-domain) and when the
            # results are already on screen (the desktop side-by-side layout).
            None, None, None,
            js="""() => {
              setTimeout(() => {
                const col = document.getElementById('results-col');
                const txt = document.querySelector(
                  '.rec-text textarea, .rec-text input');
                if (!col || !txt || !txt.value.trim()) return;
                const top = col.getBoundingClientRect().top;
                if (top >= 0 && top < window.innerHeight * 0.5) return;
                col.scrollIntoView({behavior: 'smooth', block: 'start'});
              }, 150);
            }""",
        )
    # One GPU inference at a time with a bounded waiting line (visitors see
    # their queue position); protects the server when the demo is public.
    demo.queue(default_concurrency_limit=1, max_size=10)
    # Abort oversized transfers during upload; 100 MB covers any legitimate
    # clip within the accepted duration (a 15 s 4K phone video is ~100-200 MB).
    demo.launch(max_file_size="100mb", show_api=False)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lipreading demo for the GRID conformer_ctc2 recipe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Trained Conformer-CTC checkpoint (.pt).")
    p.add_argument("--tokens", type=Path, required=True,
                   help="tokens.txt (sizes the model; used by ctc-greedy decoding).")
    p.add_argument("--method", choices=["ctc-greedy", "1best"], default="ctc-greedy",
                   help="Decoding method. ctc-greedy uses --tokens only; "
                        "1best is lexicon-constrained (needs --HLG and --words-file).")
    p.add_argument("--HLG", type=Path, default=None,
                   help="HLG.pt decoding graph (required for --method 1best).")
    p.add_argument("--words-file", type=Path, default=None,
                   help="words.txt symbol table (required for --method 1best).")
    p.add_argument("--avhubert-code-dir", type=Path, default=Path("av_hubert"),
                   help="AV-HuBERT source dir (added temporarily to sys.path).")
    p.add_argument("--avhubert-ckpt", type=Path, required=True,
                   help="AV-HuBERT pretrained checkpoint (.pt).")
    p.add_argument("--dlib-predictor", type=Path,
                   default=Path("download/dlib/shape_predictor_68_face_landmarks.dat"),
                   help="dlib 68-point landmark model.")
    p.add_argument("--layer", type=int, default=9,
                   help="AV-HuBERT encoder output layer (must match training).")
    p.add_argument("--roi-mode", choices=["centroid", "meanface"],
                   default="centroid",
                   help="Mouth-ROI geometry. 'centroid': crop around the mouth "
                        "centre, scaled by measured head size (--normalize-head). "
                        "'meanface': warp each frame onto a reference face "
                        "first (AV-HuBERT's own preprocessing), which also "
                        "corrects pose and in-plane rotation. MUST match the "
                        "geometry --checkpoint was trained on. "
                        "Default: %(default)s")
    p.add_argument("--mean-face", type=Path,
                   default=Path("download/20words_mean_face.npy"),
                   help="Reference landmarks for --roi-mode meanface. "
                        "Default: %(default)s")
    p.add_argument("--normalize-head", type=str2bool, default=True,
                   help="Scale the mouth crop to the detected face size (IOD) so "
                        "the mouth-to-head ratio matches GRID for arbitrary "
                        "videos. No-op for GRID-sized faces. Default: %(default)s")
    p.add_argument("--word-conf-calibration", type=Path, default=None,
                   help="Isotonic word-confidence map fitted for THIS "
                        "checkpoint. The map is model-specific: pairing it "
                        "with a different model silently reports wrong "
                        "confidences. Defaults to the BPE-58 map.")
    p.add_argument("--mediapipe-fallback", type=str2bool, default=True,
                   help="When dlib's HOG detector finds no face in a frame, "
                        "fall back to MediaPipe face detection (fast, robust "
                        "to glasses/tight framing). Default: %(default)s")
    # Model geometry -- must match the trained checkpoint.
    p.add_argument("--encoder-dim", type=int, default=128)
    p.add_argument("--num-encoder-layers", type=int, default=6)
    p.add_argument("--num-decoder-layers", type=int, default=3)
    p.add_argument("--ui", action="store_true",
                   help="Launch the Gradio web UI instead of CLI.")
    p.add_argument("--idle-offload-seconds", type=int, default=300,
                   help="Move the models off the GPU to CPU after this many "
                        "seconds with no request, freeing GPU memory; the next "
                        "request restores them (<1 s). 0 disables. Ignored "
                        "without a CUDA device. Default: %(default)s")
    p.add_argument("--max-per-hour", type=int, default=10,
                   help="UI quota: videos (uploaded or recorded) allowed "
                        "per visitor (IP) per hour.")
    p.add_argument("--max-per-day", type=int, default=30,
                   help="UI quota: videos (uploaded or recorded) allowed "
                        "per visitor (IP) per day.")
    p.add_argument("--activity-log", type=Path,
                   default=Path("demo_saved/activity.log"),
                   help="Append an anonymous line per UI submission "
                        "(timestamp, upload/webcam, confidence numbers only; "
                        "no content, no IP). Pass '' to disable.")
    p.add_argument("--examples", type=Path, nargs="*",
                   default=[Path("download/grid-corpus/s1/bbaf2n.mpg"),
                            Path("download/grid-corpus/s2/sgbp4s.mpg"),
                            Path("download/grid-corpus/s20/pgwj3p.mpg"),
                            Path("download/grid-corpus/s22/srwaza.mpg"),
                            Path("s3_l_lgin3a.mov"),
                            Path("recorded_demo_lgwf8a.mp4")],
                   help="Example clips offered in the UI (missing files are "
                        "skipped). Pass no paths to disable.")
    p.add_argument("video", nargs="?", default=None,
                   help="Video path (CLI mode; omit when using --ui).")
    p.add_argument("--noise", type=float, nargs="+", default=None,
                   help="Gaussian noise sigma(s) applied to the mouth ROI "
                        "before recognition (matches the UI slider, 0-50). "
                        "Multiple values re-run the same video once per "
                        "value, reusing the loaded model -- a noise/blur "
                        "robustness sweep. Paired index-wise with --blur.")
    p.add_argument("--blur", type=float, nargs="+", default=None,
                   help="Gaussian blur sigma(s) applied to the mouth ROI "
                        "before recognition (matches the UI slider, 0-5). "
                        "Same broadcasting/pairing rule as --noise.")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_args()
    reader = LipReader(args)

    if args.ui:
        launch_ui(reader, args.examples,
                  args.max_per_hour, args.max_per_day,
                  args.activity_log if str(args.activity_log) else None)
    else:
        if args.video is None:
            raise SystemExit("Provide a video path, or pass --ui for the web UI.")
        noise = args.noise or [0.0]
        blur = args.blur or [0.0]
        n = max(len(noise), len(blur))
        if len(noise) == 1 and n > 1:
            noise = noise * n
        if len(blur) == 1 and n > 1:
            blur = blur * n
        if len(noise) != len(blur):
            raise SystemExit("--noise/--blur must have matching lengths "
                              "(or a single value to broadcast)")
        for n_sigma, b_sigma in zip(noise, blur):
            text, _, word_confs, ood = reader.recognize(
                args.video, noise_sigma=n_sigma, blur_sigma=b_sigma)
            tag = f"[noise={n_sigma:g} blur={b_sigma:g}]"
            if ood == "hide":
                print(f"{tag} [out-of-domain] not a GRID sentence; no result shown")
            else:
                print(f"{tag} {text}")
                if word_confs:
                    print(" ".join(f"{w}({c:.2f})" for w, c in word_confs))


if __name__ == "__main__":
    main()
