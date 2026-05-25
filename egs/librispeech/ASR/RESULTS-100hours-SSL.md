# Results for train-clean-100

This page shows the WERs for test-clean/test-other using only
train-clean-100 subset as training data.


## Conformer encoder + embedding decoder

### 2026-05-21

| WER | reworked ctc attention | train-clean-100 | |
|------------------------|------|------|-----|
| test sets | test-clean | test-other | Avg |
| ctc-greedy-search      | 7.58% | 16.24%| |
| ctc-decoding           | 7.58% | 16.24%| |
| 1best                  | 5.30% | 11.88%| |
| nbest                  | 5.30% | 11.87%| |
| whole-lattice-rescoring| | | |
| attention-decoder      | | | |
| nbest-oracle           | 3.15% | 7.89%| |

The training command for reproducing is given below:

```bash
cd egs/librispeech/ASR/
./prepare.sh
./prepare_ssl_features.sh

export CUDA_VISIBLE_DEVICES="0"


./conformer_ctc2_ssl/train.py \
  --world-size 1 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --enable-musan 0 \
  --exp-dir conformer_ctc2_ssl/exp \
  --mini-libri 1 \
  --max-duration 300 \
  --manifest-dir data/hubert-base_9 \ 
  --spec-aug-time-warp-factor 1 \
  --bpe-model data/lang_bpe_500/bpe.model 

```

The decoding command is given below:

```bash
for method in ctc-greedy-search ctc-decoding 1best nbest-oracle; do
    ./conformer_ctc2_ssl/decode.py \
      --exp-dir conformer_ctc2_ssl/exp16 \
      --epoch 30 \
      --avg 15 \
      --method 1best \
      --manifest-dir data/hubert-base_9 
    
done
