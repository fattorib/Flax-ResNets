# ResNets: Jax/Flax vs. PyTorch 

I want to learn JAX/Flax by implementing something (slightly) non-trivial and comparing it against an equivalent PyTorch implementation.

Reference: PyTorch (1 year exp), Jax (No experience)

## Notes:
- Flax doesn't have a Sequential class. See (https://github.com/google/flax/pull/1156). Not that hard to write yourself
- Calling BatchNorm is weird (train/eval changes, params + variables)