# ResNets: Jax/Flax vs. PyTorch 

I want to learn JAX/Flax by implementing something (slightly) non-trivial and comparing it against an equivalent PyTorch implementation.

Reference: PyTorch (1 year exp), Jax (No experience)

# 1. Model Construction

# 2. Data Loading

In JAX/Flax, we can actually take the existing PyTorch data pipeline and modify it slightly to return jnp arrays instead of PyTorch Tensors. See [here](https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html) for more details. PyTorch's data loading and augmentations capabilities are great so being able to directly use this with Flax is great.

One issue I noticed was that my code would always return an OOM error if I set ```pin_memory=True``` in the dataloader. I suspect this is because JAX, by default will allocate tensors directly to the GPU memory, instead of the pinned memory. 

# 3. Model Training

## 3.1 TrainState
In Flax, all model training is passed through a ```TrainState``` class which holds the ```.apply()``` method, the optimizer, the model paramaters and any other attributes we wish to include. In the Flax example, I have created a subclass of ```TrainState``` and included batch statistics, weight decay, and dynamic scaling as extra attributes. 

## 3.2 Optimizers
The Flax docs recommend using [Optax](https://github.com/deepmind/optax) for optimizers and learning rate scheduling. By default, only the AdamW optimizer includes a weight decay parameter.

Weight decay/L2 regularization can get a bit tricky depending on the optimizer used (See Adam vs. AdamW). In our case, with SGD, we can add an L2 regularization term manually to our loss function. It is common practice to exclude certain paramaters from regulurization, including $\alpha$ and $\beta$ in BatchNorm layers and bias terms in Dense\Linear layers. 

In PyTorch, we can filter these paramaters through ```model.named_parameters()```:
```python

#Taken from timm (https://github.com/rwightman/pytorch-image-models)
for key, value in model.named_parameters():

if "fc.bias" in key or "bias" in key or "bn" in key:
    #exclude params for weight decay
else:
    #include params for weight decay
```

In Flax, we can use the following section of code:

```python 
weight_decay_params_filter = flax.traverse_util.ModelParamTraversal(
        lambda path, _: ("bias" not in path and "scale" not in path)
)

weight_decay_params = weight_decay_params_filter.iterate(params)

```

Adding a learning rate schedule is quite easy. Optax supports many of the common ones. Since the schdule is passed in as a function to the optimizer, all lr steps are handled internally compared with PyTorch which requires calling ```scheduler.step()```. 

# 4. Miscellaneous 




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

- Default Flax train state is based on steps, not epochs

- Weight decay is whack (I think)

- Mixed Precision anyone?
