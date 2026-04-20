## Results

### [conformer_ctc2](./conformer_ctc2)

### 2026-04-18


| WER | |
|------------------------|-----|
| ctc-greedy-search      | 7.79% |
| ctc-decoding           | 7.79% |
| 1best                  | 6.94% |


The training command using 1 NVIDIA GeForce RTX 3080 GPU is:
```bash
export CUDA_VISIBLE_DEVICES="0,1"
# for non-streaming model training:
./conformer_ctc2/train.py \
  --max-duration 1400
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in ; do
  ./conformer_ctc2/decode.py \
    --epoch 28 \
    --avg 20 \
    --max-duration 1400 \
    --decoding-method $m \
    --use-averaged-model False
done
```

