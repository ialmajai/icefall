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
from pathlib import Path

import numpy as np
import torch

from conformer import Conformer
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
# Accepted clip length. Below the minimum no GRID-style sentence fits; above
# the maximum, per-frame landmarking would tie up the demo for too long.
# The tolerance keeps nominally-3s clips (e.g. 74-frame GRID clips, 2.96s)
# from being rejected over metadata rounding.
MIN_DURATION_S = 3.0
MAX_DURATION_S = 30.0
DURATION_TOL_S = 0.1

# Head-size normalization. GRID has uniform framing: its fixed 64px mouth crop
# corresponds to a median interocular distance (IOD) of ~50.6px. For arbitrary
# videos the face can be any size, so we scale the crop to the measured IOD
# (crop = CROP_PER_IOD * IOD) before resizing to ROI_SIZE, keeping the
# mouth-to-head ratio consistent with GRID. For a GRID-sized face this yields
# ~64px (unchanged), so normalization is a safe generalization.
GRID_IOD_REF = 50.6
CROP_PER_IOD = MOUTH_W / GRID_IOD_REF  # ~1.264


def _median_iod(landmarks: np.ndarray) -> float:
    """Median interocular distance (px) across frames from dlib 68 landmarks."""
    lm = landmarks.astype(float)
    reye = lm[:, 36:42].mean(axis=1)  # right-eye centre per frame
    leye = lm[:, 42:48].mean(axis=1)  # left-eye centre per frame
    return float(np.median(np.linalg.norm(leye - reye, axis=1)))

# Shown in the UI so users know what a valid GRID sentence looks like.
GRID_GRAMMAR_MD = """
### Valid GRID sentence structure

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

# Shown in the UI (GDPR Art. 13 information for volunteers).
PRIVACY_NOTICE_MD = """
### Privacy notice

**Who is collecting**: Ibrahim Almajai, independent researcher
(i.almajai@gmail.com).

**What & why**: If — and only if — you tick the consent box, two things are
saved on the server: the cropped mouth-region frames extracted from your clip,
and the clip's audio track. They are used for lipreading (VSR) and
audio-visual speech recognition research, where the audio provides
ground-truth supervision and alignment for the visual data. The full video is
never stored; uploaded clips are processed in temporary files that are
routinely deleted. Be aware that a voice recording may identify you.

**Record alone**: please record by yourself in a quiet room — the microphone
also captures other people's voices, and they cannot consent through this
form.

**Retention**: Saved data is deleted at most 12 months after collection.

**Your rights**: You can withdraw consent and have your data deleted at any
time: email the *Saved data ID* shown after submitting to the address above.
You may also request a copy of the data, and you have the right to complain
to your data protection authority.

This demo is intended for adults (18+).
"""

# HLG lattice-decoding hyper-parameters (from decode.py get_params()).
SEARCH_BEAM = 20
OUTPUT_BEAM = 4
MIN_ACTIVE_STATES = 30
MAX_ACTIVE_STATES = 10000


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
        self.avhubert = g["model"]
        self.transform = g["transform"]
        self.device = g["device"]
        self.layer = args.layer
        self.normalize_head = args.normalize_head
        self.save_dir = args.save_dir

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

    def _roi_and_features(self, video: str):
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
                video, Path(resampled) if resampled else Path(video)
            )
        finally:
            if resampled is not None:
                Path(resampled).unlink(missing_ok=True)

    def _roi_and_features_25fps(self, video: str, video_path: Path):
        """`_roi_and_features` body; video_path is the (possibly resampled)
        clip to process, video the original name for messages."""
        landmarks = _detect_landmarks(video_path, self.detector, self.predictor, 1)
        if landmarks is None:
            raise RuntimeError(f"No face detected in {video}")
        # Head-size normalization: scale the mouth crop to the detected face
        # size (via IOD) so the mouth-to-head ratio matches GRID regardless of
        # how big the face is in the frame. On GRID this yields ~64px (no-op).
        #
        # TODO: normalize head *pose*, not just scale -- align each frame to a
        # reference/mean face via a similarity transform on stable landmarks
        # (eyes/nose), as in AV-HuBERT's align_mouth. This would correct
        # in-plane rotation and non-frontal pose before cropping, instead of the
        # current scale-only, per-video-median-IOD approximation.
        if self.normalize_head:
            crop = max(16, round(CROP_PER_IOD * _median_iod(landmarks)))
            logging.info(f"Head-normalized mouth crop: {crop}px "
                         f"(IOD {_median_iod(landmarks):.1f}px)")
        else:
            crop = MOUTH_W
        raw = _extract_mouth_frames(
            video_path, landmarks, crop, crop, ROI_SIZE, MOUTH_LEFT, MOUTH_RIGHT
        )
        roi = np.stack(raw)  # (T, 88, 88) uint8
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
        from icefall.decode import get_lattice, one_best_decoding
        from icefall.utils import get_texts

        T = nnet_output.shape[1]
        # Single, unpadded clip: one segment covering all T frames.
        supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
        lattice = get_lattice(
            nnet_output=nnet_output,
            decoding_graph=self.HLG,
            supervision_segments=supervision_segments,
            search_beam=SEARCH_BEAM,
            output_beam=OUTPUT_BEAM,
            min_active_states=MIN_ACTIVE_STATES,
            max_active_states=MAX_ACTIVE_STATES,
            subsampling_factor=1,
        )
        best_path = one_best_decoding(lattice=lattice, use_double_scores=True)
        hyp = get_texts(best_path)[0]
        words = [self.word_table[i] for i in hyp]
        word_confs = self._word_confidences_1best(best_path, nnet_output, words)
        return " ".join(words), word_confs

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
    def recognize(self, video: str):
        """Return (recognised_text, raw ROI frames, word_confs).

        word_confs is a list of (word, confidence in [0, 1]) pairs; it may be
        empty when confidences could not be derived."""
        duration = _video_duration(video)
        if duration and not (
            MIN_DURATION_S - DURATION_TOL_S
            <= duration
            <= MAX_DURATION_S + DURATION_TOL_S
        ):
            raise ValueError(
                f"Clip is {duration:.1f}s long; please use a clip between "
                f"{MIN_DURATION_S:g}s and {MAX_DURATION_S:g}s."
            )
        roi, feats = self._roi_and_features(video)
        feature = feats.unsqueeze(0).to(self.device)  # (1, T, 768)
        nnet_output = self.model(feature, None)[0]  # (1, T, C)
        if self.method == "1best":
            text, word_confs = self._decode_1best(nnet_output)
        else:
            text, word_confs = ctc_greedy_decode(nnet_output.cpu(), self.id2sym)
        return text, roi, word_confs


def _video_fps(video: str) -> float:
    """Container frame rate as reported by OpenCV (0.0 if unknown)."""
    import cv2

    cap = cv2.VideoCapture(video)
    try:
        return cap.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        cap.release()


def _video_duration(video: str) -> float:
    """Duration in seconds from container metadata (0.0 if unknown)."""
    import cv2

    cap = cv2.VideoCapture(video)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        return frames / fps if fps > 0 and frames > 0 else 0.0
    finally:
        cap.release()


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


def _frames_to_gif(frames: np.ndarray, fps: int = 25, scale: int = 3, out_dir=None):
    """Encode grayscale ROI frames (T,H,W uint8) to an animated GIF (Pillow),
    nearest-neighbour upscaled by `scale`. Returns the path to a temp .gif."""
    import tempfile

    from PIL import Image

    _, H, W = frames.shape
    imgs = [
        Image.fromarray(f, mode="L").resize((W * scale, H * scale), Image.NEAREST)
        for f in frames
    ]
    out = tempfile.NamedTemporaryFile(suffix=".gif", delete=False, dir=out_dir).name
    imgs[0].save(
        out, save_all=True, append_images=imgs[1:],
        duration=int(1000 / fps), loop=0,
    )
    return out


def _save_roi_data(roi: np.ndarray, video: str, save_dir: Path) -> str:
    """Save the extracted mouth-ROI frames (T,H,W uint8) as a timestamped .npz
    plus the clip's audio track as a .wav with the same stem. The full face
    video is never stored. Note the audio is voice, i.e. potentially
    identifying data -- both files are saved only with the user's explicit
    consent, and the shared stem is the ID users quote to request deletion.
    Returns that ID."""
    import shutil
    import subprocess
    from datetime import datetime

    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stem = f"roi_{stamp}"
    np.savez_compressed(save_dir / f"{stem}.npz", frames=roi.astype(np.uint8))
    wav = save_dir / f"{stem}.wav"
    if shutil.which("ffmpeg") is None:
        logging.warning(f"ffmpeg unavailable; audio track of {stem} not saved.")
    else:
        p = subprocess.run(
            ["ffmpeg", "-y", "-i", video, "-vn", "-acodec", "pcm_s16le", str(wav)],
            capture_output=True,
        )
        if p.returncode != 0:  # e.g. the clip has no audio track
            wav.unlink(missing_ok=True)
            logging.warning(f"No audio saved for {stem} (no track/ffmpeg failed).")
    logging.info(f"Saved consented data with ID {stem} to {save_dir}")
    return stem


def launch_ui(reader: LipReader, example_paths=None):
    import gradio as gr

    def infer(video, consent):
        if not video:
            return None, "", None, None, None, None, ""
        try:
            text, roi, word_confs = reader.recognize(video)
        except Exception as e:
            logging.exception(f"Recognition failed for {video}")
            raise gr.Error(f"Recognition failed: {e}")
        saved_id = ""
        if consent:
            saved_id = _save_roi_data(roi, video, reader.save_dir)
        out_dir = _new_request_dir()
        return (
            _to_playable_mp4(video, out_dir),
            text,
            # Score in the text (gradio only tints, never prints, labels) plus
            # a bucket category for the colour: float labels tint red like a
            # saliency map, so use color_map'd buckets instead.
            [
                (f"{w} {c:.2f}",
                 "high" if c >= 0.9 else "medium" if c >= 0.7 else "low")
                for w, c in word_confs
            ] or None,
            _frames_to_mp4(roi, out_dir=out_dir),
            # GIF slower than real-time, easier to read.
            _frames_to_gif(roi, fps=10, out_dir=out_dir),
            roi_strip(roi),
            saved_id,
        )

    # Example clips: GRID .mpg won't play in the browser, so show a transcoded
    # mp4 copy (kept for the server's lifetime) when ffmpeg is available.
    examples = []
    for pth in example_paths or []:
        if not Path(pth).exists():
            logging.warning(f"Example clip not found, skipping: {pth}")
            continue
        examples.append([_to_playable_mp4(str(pth)) or str(pth)])

    with gr.Blocks(title="GRID Lipreading (VSR) Demo") as demo:
        gr.Markdown("# GRID Lipreading (VSR) Demo")
        gr.Markdown(
            "AV-HuBERT visual features → Conformer-CTC. "
            "Upload a frontal talking-face video "
            f"({MIN_DURATION_S:g}–{MAX_DURATION_S:g} s)."
        )
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="Frontal talking-face clip")
                save_cb = gr.Checkbox(
                    label="I consent to the cropped mouth-region frames and "
                          "the clip's audio track being saved on the server "
                          "for lipreading and audio-visual speech research. "
                          "The full video is not stored; note that a voice "
                          "recording may be identifying.",
                    value=False,
                )
                saved_out = gr.Textbox(
                    label="Saved data ID (quote this to request deletion)",
                    interactive=False,
                )
                with gr.Accordion("Privacy notice", open=False):
                    gr.Markdown(PRIVACY_NOTICE_MD)
                submit = gr.Button("Submit", variant="primary")
                if examples:
                    gr.Examples(
                        examples=examples, inputs=[video_in],
                        label="Example clips (click, then Submit)",
                    )
                # Static, always visible (before and after submit), under submit.
                gr.Markdown(GRID_GRAMMAR_MD)
            with gr.Column():
                playback = gr.Video(label="Playback (transcoded to mp4)")
                text_out = gr.Textbox(label="Recognised text")
                conf_out = gr.HighlightedText(
                    label="Word confidence (CTC posterior)",
                    color_map={"high": "green", "medium": "yellow", "low": "red"},
                    show_inline_category=False,
                    show_legend=False,
                )
                roi_vid = gr.Video(
                    label="Mouth ROI (animated)", show_download_button=True,
                    height=200, width=200,
                )
                gif_out = gr.File(label="Download ROI animation (GIF)")
                strip_out = gr.Image(
                    label="Mouth ROI (sampled frames)", image_mode="L"
                )
        submit.click(
            infer,
            inputs=[video_in, save_cb],
            outputs=[playback, text_out, conf_out, roi_vid, gif_out, strip_out,
                     saved_out],
        )
    demo.launch()


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
    p.add_argument("--normalize-head", type=str2bool, default=True,
                   help="Scale the mouth crop to the detected face size (IOD) so "
                        "the mouth-to-head ratio matches GRID for arbitrary "
                        "videos. No-op for GRID-sized faces. Default: %(default)s")
    p.add_argument("--save-dir", type=Path, default=Path("demo_saved"),
                   help="Server-side dir where mouth-ROI data (.npz) and the "
                        "clip's audio track (.wav) are saved when the user "
                        "ticks the consent box. The full video is never "
                        "stored. Default: %(default)s")
    # Model geometry -- must match the trained checkpoint.
    p.add_argument("--encoder-dim", type=int, default=128)
    p.add_argument("--num-encoder-layers", type=int, default=6)
    p.add_argument("--num-decoder-layers", type=int, default=3)
    p.add_argument("--ui", action="store_true",
                   help="Launch the Gradio web UI instead of CLI.")
    p.add_argument("--examples", type=Path, nargs="*",
                   default=[Path("grid-corpus/s1/bbaf2n.mpg"),
                            Path("grid-corpus/s33/bbac1s.mpg")],
                   help="Example clips offered in the UI (missing files are "
                        "skipped). Pass no paths to disable.")
    p.add_argument("video", nargs="?", default=None,
                   help="Video path (CLI mode; omit when using --ui).")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_args()
    reader = LipReader(args)

    if args.ui:
        launch_ui(reader, args.examples)
    else:
        if args.video is None:
            raise SystemExit("Provide a video path, or pass --ui for the web UI.")
        text, _, word_confs = reader.recognize(args.video)
        print(text)
        if word_confs:
            print(" ".join(f"{w}({c:.2f})" for w, c in word_confs))


if __name__ == "__main__":
    main()
