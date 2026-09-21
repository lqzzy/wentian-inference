# Contributing

Keep public interfaces small and preserve numerical behavior. Generated
predictions, Slurm logs, local data, caches, and alternate checkpoints must not
be committed.

Before submitting a change, run:

```bash
make test
make verify
make lint
```

Changes to model operators, spatial decomposition, memory placement, precision
conversion, or worker synchronization should also be checked on a full 608-core
Kunpeng 920F node in both FP32 and FP64.
