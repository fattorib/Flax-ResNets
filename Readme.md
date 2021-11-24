# ResNets: Jax/Flax vs. PyTorch 

I want to learn JAX/Flax by implementing something (slightly) non-trivial and comparing it against an equivalent PyTorch implementation.

Reference: PyTorch (1 year exp), Jax (No experience)

## Notes:
- Flax doesn't have a Sequential class. See (https://github.com/google/flax/pull/1156). Not that hard to write yourself
- Calling BatchNorm is weird (train/eval changes, params + variables)

- For mode switching behaviour (Dropout/BN), the switch must be passed to the __call__ method 

- XLA is more finicky than CUDA, had to update CUDA

- JAX defaults to different indice ordering for Convolutions

- With a few small changes, dataloading can be handled via pytorch  (unless you want to pin memory! Then its weird)

- JAX jnp arrays by default will be placed on the default device (https://jax.readthedocs.io/en/latest/faq.html). Using pinned memory in PyTorch dataloaders will result in all the pinned data being placed on the GPU automatically - easy to arrive at OOM errors. One workaround is to have the dataloader return np arrays and convert them to jnp after. Another solution is to not used Pinned memory 

- OOM errors are odd

- Not super clear how GPU use works

- Any JIT compliation will not like conditional stataments