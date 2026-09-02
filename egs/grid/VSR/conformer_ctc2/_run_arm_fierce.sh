#!/bin/bash
# Generic arm runner: train -> in-domain decode -> Lombard eval, all with the
# same settings so train/test can never silently mismatch.
#
#   NAME=x EPOCHS=12 DROPOUT=0.4 CMN=none ENCDIM=128 ATT=0.5 \
#       bash conformer_ctc2/_run_arm_fierce.sh
#
# Writes to /srv/nfs/data/vsr-exp/$NAME (local disk on fierce, keeps /data free).
#
# set -e is load-bearing: without it a failed train step still ran the decode
# and the Lombard eval, which happily scored whatever checkpoints were already
# in $EXP and printed a plausible number under the new arm's name.
set -euo pipefail
cd /data/icefall/egs/grid/VSR
: "${NAME:?need NAME}"
EPOCHS="${EPOCHS:-12}"
DROPOUT="${DROPOUT:-0.1}"
CMN="${CMN:-none}"
ENCDIM="${ENCDIM:-128}"
ATT="${ATT:-0.5}"
DROPSCHED="${DROPSCHED:-}"
MANIFEST="${MANIFEST:-data/avhubert_mf_l4}"
UPSAMPLE="${UPSAMPLE:-1}"
# Cuts with a doubled nominal duration halve the batch at a fixed --max-duration,
# so 50 fps arms pass MAXDUR=768 to keep the batch size comparable.
MAXDUR="${MAXDUR:-384}"
SPKADV="${SPKADV:-0}"   # gradient-reversal speaker-adversarial weight
SEED="${SEED:-42}"     # was missing: SEED= was silently ignored, so runs
                       # labelled s43/s44 were all seed 42 repeats
# LANG is the POSIX locale variable. Launched from any shell that has it set
# (every interactive one), "${LANG:-data/lang_bpe_58}" resolved to en_GB.UTF-8
# and the run died on en_GB.UTF-8/tokens.txt. It only ever worked because the
# arms were launched over `ssh fierce "cmd"`, where LANG is unset. Prefer
# LANGDIR (what _run_ablation_arm.sh already uses); still accept a LANG that
# looks like a lang dir, for the existing callers that pass it that way.
LANGDIR="${LANGDIR:-}"
if [ -z "$LANGDIR" ]; then
  case "${LANG:-}" in
    data/*|/*) LANGDIR="$LANG" ;;
    *)         LANGDIR="data/lang_bpe_58" ;;
  esac
fi
TOKENS="${TOKENS:-}"   # phone models need lang_phone/tokens_model.txt
                       # (37 units + <sos/eos> = 38 classes); empty = use <LANG>/tokens.txt   # data/lang_phone selects the phone lexicon
LAYER="${LAYER:-4}"     # AV-HuBERT layer the features came from; the
                        # Lombard eval recomputes features from ROIs and
                        # must use the same layer or it silently mismatches
AVG="${AVG:-$((EPOCHS / 2))}"
PY=/data/miniconda3/envs/icefall-vsr/bin/python
EXP=/srv/nfs/data/vsr-exp/$NAME

echo "=== arm $NAME: epochs=$EPOCHS dropout=$DROPOUT sched='$DROPSCHED' cmn=$CMN dim=$ENCDIM att=$ATT avg=$AVG"
$PY conformer_ctc2/train.py \
    --exp-dir "$EXP" \
    --manifest-dir "$MANIFEST" \
    --lang-dir "$LANGDIR" \
    --max-duration "$MAXDUR" --num-epochs "$EPOCHS" --start-epoch 1 \
    --num-workers 2 --world-size 1 \
    --on-the-fly-feats false --enable-spec-aug false \
    --dropout "$DROPOUT" --cmn "$CMN" --encoder-dim "$ENCDIM" --att-rate "$ATT" \
    --seed "$SEED" \
    --speaker-adv-weight "$SPKADV" \
    --dropout-schedule "$DROPSCHED"

$PY conformer_ctc2/decode.py \
    --exp-dir "$EXP" \
    --manifest-dir "$MANIFEST" \
    --lang-dir "$LANGDIR" \
    --method 1best --epoch "$EPOCHS" --avg "$AVG" \
    --max-duration "$MAXDUR" --on-the-fly-feats false --enable-spec-aug false \
    --cmn "$CMN" --encoder-dim "$ENCDIM"

$PY conformer_ctc2/_lombard_eval.py \
    --exp-dir "$EXP" --epoch "$EPOCHS" --avg "$AVG" --lang-dir "$LANGDIR" \
    --tokens "$TOKENS" \
    --cmn "$CMN" --encoder-dim "$ENCDIM" --upsample "$UPSAMPLE" \
    --layer "$LAYER" --tag "$NAME"
echo "=== arm $NAME done"
