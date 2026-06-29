# BPC
Background Plain classification.

## Train

```bash
SIZE=48x48
VAR=p
uv run python -m bpc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/bpc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 1 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=n
uv run python -m bpc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/bpc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 2 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=t
uv run python -m bpc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/bpc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 3 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=s
uv run python -m bpc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/bpc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=l
uv run python -m bpc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/bpc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 8 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```

## Export ONNX

Training exports the best checkpoint to ONNX automatically.
To export a checkpoint manually, run:

```bash
uv run python -m bpc exportonnx \
--checkpoint runs/bpc_is_l_48x48/bpc_best_epoch0067_f1_0.9430.pt \
--output bpc_is_l_48x48.onnx \
--opset 17 \
--device cpu
```

Use the `bpc_best_*.pt` checkpoint from the target run directory.
