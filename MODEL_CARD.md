# Wentian model card

## Model

Wentian is a global weather forecasting network with 3D patch embedding,
windowed transformer blocks, down/up sampling, skip connections, and separate
pressure-level and surface recovery heads.

| Item | Value |
|---|---|
| Grid | 721 × 1440 |
| Pressure levels | 13 |
| Input history | 2 × 6 hours |
| Pressure variables | 5 input / 5 output |
| Surface variables | 7 input / 4 output |
| Embedding dimension | 256 |
| Transformer depths | `[2, 6, 6, 2]` |
| Attention heads | `[2, 8, 8, 2]` |
| Patch size | `[2, 4, 4]` |
| Window size | `[2, 6, 12]` |
| Supported precision | FP32 and FP64 |

## Checkpoint

```text
archive:        weights/wentian_beta.pth.gz
archive size:   274,911,774 bytes
archive sha256: 9071dbe34001aa620cfa1378e1870d1d35960fa426ead6d7ba07ff5cbb461a3b
restored size:  1,382,298,257 bytes
restored sha256:56b68db5ae3b64e698bcccce1b552dc4a60caa0c0113d1372365b0c9a6198120
storage:        Git LFS
```

The same trained checkpoint is converted to the selected runtime precision at
load time. FP64 therefore evaluates the same learned parameters; it does not
add information that was absent from the checkpoint.

## Inputs and outputs

The runner loads two consecutive ERA5 states, selects and normalizes the model
variables, then performs 60 autoregressive 6-hour steps. Each output file stores
physical pressure-level and surface tensors plus the timestamp, step, precision,
lead time, and forward wall time.

## Intended use and limitations

This release supports research inference and performance studies on Kunpeng
920F systems. It is not an operational forecast service. Forecast quality
depends on matching the expected ERA5 variable order, levels, grid, units, and
normalization constants. Users should independently validate forecasts before
using them in scientific or safety-relevant decisions.
