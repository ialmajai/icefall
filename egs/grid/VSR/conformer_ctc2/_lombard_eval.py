#!/usr/bin/env python3
"""Lombard GRID eval of any conformer_ctc2 arm, on the cached mean-face ROIs.

Parameterised so every experiment is scored the same way instead of by a
hand-edited copy: same 2,577 clips ssl-kaldi scores, same HLG 1-best decoding,
same references (derived from the filename code, verified identical to their
transcripts).

  python conformer_ctc2/_lombard_eval.py --exp-dir conformer_ctc2/exp-mf-l4 \
      --epoch 30 --avg 15 [--encoder-dim 128] [--cmn none] [--layer 4]
      [--min-frames 30] [--tag name]

Prints WER over all scored clips and over the 2,514-clip subset used by the
earlier arms, so numbers stay comparable across the whole investigation.
Throwaway.
"""
import argparse
import glob
import re
import sys
import time
from pathlib import Path

sys.argv = sys.argv + ["_"]  # av_hubert argv-less DBG guard

import numpy as np  # noqa: E402
import torch  # noqa: E402
import k2  # noqa: E402

sys.path.insert(0, "conformer_ctc2")
sys.path.insert(0, "local")
import demo  # noqa: E402
from cmn import apply_cmn  # noqa: E402

cli = argparse.ArgumentParser()
cli.add_argument("--exp-dir", required=True)
cli.add_argument("--epoch", type=int, default=30)
cli.add_argument("--avg", type=int, default=15)
cli.add_argument("--encoder-dim", type=int, default=128)
cli.add_argument("--cmn", choices=["none", "utt"], default="none")
cli.add_argument("--layer", type=int, default=4)
cli.add_argument("--min-frames", type=int, default=30)
cli.add_argument("--tokens", default="",
                 help="tokens.txt to size the model from; defaults to "
                      "<lang-dir>/tokens.txt. Use lang_phone/tokens_model.txt "
                      "for phone models, which allocate an extra <sos/eos> class.")
cli.add_argument("--num-classes", type=int, default=0,
                 help="override the model's output size. The phone lexicon has "
                      "37 tokens plus a dedicated <sos/eos> = 38 classes, but "
                      "tokens.txt only lists the 37, so LipReader infers the "
                      "wrong size and the checkpoint fails to load.")
cli.add_argument("--lang-dir", default="data/lang_bpe_58",
                 help="lexicon dir; data/lang_phone uses the phone units "
                      "instead of the 58 BPE units (different tokens.txt, "
                      "HLG.pt and words.txt)")
cli.add_argument("--roi-mode", choices=["meanface", "centroid"], default="meanface",
                 help="ROI geometry the checkpoint was trained on. centroid ROIs are not pre-cached, so they are extracted once via the demo pipeline and cached as <clip>.cenroi.npz")
cli.add_argument("--fps-resample", type=str, default="false",
                 help="resample the cached ROI sequence to 25 fps. Lombard "
                      "video is ~24 fps while GRID/AV-HuBERT is 25, and the "
                      "cached ROIs are raw frames, so without this the model "
                      "sees speech ~4%% slow. The demo path (_roi_and_features) "
                      "always resamples; the caches bypass it.")
cli.add_argument("--upsample", type=int, default=1,
                 help="repeat each feature frame N times, to match a model trained on upsampled (e.g. 50 fps) features")
cli.add_argument("--limit", type=int, default=0,
                 help="score only the first N clips (smoke tests)")
cli.add_argument("--tag", default=None)
opts = cli.parse_args([a for a in sys.argv[1:] if a != "_"])
tag = opts.tag or Path(opts.exp_dir).name

CMD = {"b": "bin", "l": "lay", "p": "place", "s": "set"}
COL = {"b": "blue", "g": "green", "r": "red", "w": "white"}
PREP = {"a": "at", "b": "by", "i": "in", "w": "with"}
DIG = {"z": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
       "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
ADV = {"a": "again", "n": "now", "p": "please", "s": "soon"}


def truth(code):
    return [CMD[code[0]], COL[code[1]], PREP[code[2]], code[3],
            DIG[code[4]], ADV[code[5]]]


def wer_counts(ref, hyp):
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    return d[n][m], n


exp = Path(opts.exp_dir)
avg_ckpt = exp / f"pretrained-e{opts.epoch}-avg{opts.avg}.pt"
src_start = exp / f"epoch-{opts.epoch - opts.avg}.pt"
src_end = exp / f"epoch-{opts.epoch}.pt"
# Rebuild whenever a source epoch checkpoint is newer than the average. An arm
# re-run into the same $EXP used to silently reuse the PREVIOUS run's average
# and report the old model's WER under the new arm's name.
if avg_ckpt.exists():
    stale = [f for f in (src_start, src_end)
             if f.exists() and f.stat().st_mtime > avg_ckpt.stat().st_mtime]
    if stale:
        print(f"stale {avg_ckpt.name}: {[f.name for f in stale]} are newer; "
              "rebuilding", flush=True)
        avg_ckpt.unlink()
if not avg_ckpt.exists():
    from icefall.checkpoint import average_checkpoints_with_averaged_model
    for f in (src_start, src_end):
        if not f.exists():
            raise SystemExit(f"missing source checkpoint for averaging: {f}")
    state = average_checkpoints_with_averaged_model(
        filename_start=str(src_start),
        filename_end=str(src_end),
        device=torch.device("cpu"),
    )
    torch.save({"model": state}, avg_ckpt)
    print(f"wrote {avg_ckpt}", flush=True)

args = argparse.Namespace(
    checkpoint=avg_ckpt,
    tokens=Path(opts.tokens) if opts.tokens else Path(opts.lang_dir) / "tokens.txt",
    method="1best",
    HLG=Path(opts.lang_dir) / "HLG.pt",
    words_file=Path(opts.lang_dir) / "words.txt",
    avhubert_code_dir=Path("av_hubert"),
    avhubert_ckpt=Path("download/avhubert-ckpts/base_vox_iter5.pt"),
    dlib_predictor=Path("download/dlib/shape_predictor_68_face_landmarks.dat"),
    layer=opts.layer, normalize_head=True, mediapipe_fallback=False,
    cmn=opts.cmn, roi_mode=opts.roi_mode,
    mean_face=Path("download/20words_mean_face.npy"),
    encoder_dim=opts.encoder_dim, num_encoder_layers=6, num_decoder_layers=3,
)
reader = demo.LipReader(args)
if opts.num_classes:
    from conformer import Conformer
    from icefall.checkpoint import load_checkpoint
    m = Conformer(
        num_features=768, num_classes=opts.num_classes, subsampling_factor=1,
        d_model=opts.encoder_dim, nhead=8, dim_feedforward=1024,
        num_encoder_layers=6, num_decoder_layers=3,
    )
    load_checkpoint(str(avg_ckpt), m)
    reader.model = m.to(reader.device).eval()
    print(f"[{tag}] rebuilt model with num_classes={opts.num_classes}", flush=True)
print(f"[{tag}] model {avg_ckpt} | layer {opts.layer} | dim {opts.encoder_dim} "
      f"| cmn {opts.cmn}", flush=True)

pat = re.compile(r"(s\d+)_([lp])_([a-z0-9]{6})$")
cached = sorted(glob.glob("lombard-grid/lombardgrid/front/*.mfroi.npz"))
if opts.limit:
    cached = cached[:opts.limit]
out_path = exp / f"lombard_hyps_{tag}.tsv"

err = n_all = 0
err50 = n50 = 0          # the >=50-frame subset the earlier arms scored
t0 = time.time()
with open(out_path, "w") as out:
    print("speaker\tcond\tcode\tref\thyp\tframes", file=out)
    for i, p in enumerate(cached, 1):
        m = pat.match(Path(p).stem.replace(".mfroi", ""))
        if not m:
            continue
        spk, cond, code = m.groups()
        if opts.roi_mode == "meanface":
            _d = np.load(p)
            roi = _d["roi"]
            if opts.fps_resample.lower() == "true":
                src_fps = float(_d["fps"]) if "fps" in _d else 25.0
                if abs(src_fps - 25.0) > 0.1:
                    # Nearest-index resample, matching what ffmpeg's fps filter
                    # does at the video level in the demo path.
                    n_out = int(round(roi.shape[0] * 25.0 / src_fps))
                    idx = np.clip(np.round(np.arange(n_out) * src_fps / 25.0
                                           ).astype(int), 0, roi.shape[0] - 1)
                    roi = roi[idx]
        else:
            # Centroid geometry: no pre-existing cache, so extract once with
            # the demo's own pipeline (dlib, ~2.3 s/clip) and cache it so a
            # second model's eval is fast.
            cen = Path(str(p).replace(".mfroi.npz", ".cenroi.npz"))
            if cen.exists():
                roi = np.load(cen)["roi"]
            else:
                vid = str(p).replace(".mfroi.npz", ".mov")
                try:
                    roi, _ = reader._roi_and_features(vid)
                except Exception as e:
                    print(f"  skip {Path(vid).name}: {e}", flush=True)
                    continue
                np.savez_compressed(cen, roi=roi)
        if roi.shape[0] < opts.min_frames:
            continue
        _, feats = reader._features_from_roi(roi)
        if opts.upsample > 1:
            feats = feats.repeat_interleave(opts.upsample, dim=0)
        feature = apply_cmn(feats.unsqueeze(0).to(reader.device), opts.cmn)
        with torch.no_grad():
            nnet_output = reader.model(feature, None)[0]
        text, _, _ = reader._decode_1best(nnet_output)
        ref = truth(code)
        e, n = wer_counts(ref, text.split())
        err += e
        n_all += n
        if roi.shape[0] >= 50:
            err50 += e
            n50 += n
        print(f"{spk}\t{cond}\t{code}\t{' '.join(ref)}\t{text}\t{roi.shape[0]}",
              file=out, flush=True)
        if i % 500 == 0:
            print(f"  {i}/{len(cached)} WER {100*err/max(n_all,1):.2f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)

print(f"\nRESULT {tag}: all clips WER {100*err/n_all:.2f}% ({err}/{n_all})")
print(f"RESULT {tag}: >=50-frame subset WER {100*err50/n50:.2f}% ({err50}/{n50})")
print(f"hyps: {out_path}")
