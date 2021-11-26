from typing import Sequence
from torch.utils import data
import numpy as np
import jax.numpy as jnp


# Utils used for training flax ResNet. Currently holds dataloader class

# From jax example
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.array(pic.permute(1, 2, 0), dtype=jnp.float32)



def create_cos_anneal_schedule(base_lr, min_lr, max_steps):
  
  def learning_rate_fn(step):
      lr = min_lr  + (0.5)*(base_lr - min_lr)*(1+jnp.cos(jnp.pi*step/max_steps))
      return lr 

  return learning_rate_fn