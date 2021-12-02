# ResNets: Jax/Flax vs. PyTorch 

I want to learn JAX/Flax by implementing something (slightly) non-trivial and comparing it against an equivalent PyTorch implementation.

Reference: PyTorch (1 year exp), Jax (No experience)

# Contents
- [Model Construction](#model-construction)
- [Data Loading](#data-loading)
- [Model Training](#model-training)
- [Other Helpful resources](#other-helpful-resources)

# Model Construction

## Defining Modules

Through Flax's Linen API, we should be able to define modules with the ```@nn.compact``` decorator. I found writing modules this way to be very simple! For a basic residual block in Flax, we would write:
```python
class ResidualBlock(nn.Module):
    # Define collection of datafields here
    in_channels: int

    # For batchnorm, we can pass it as a ModuleDef
    norm: ModuleDef

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv layers
    kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x):
        residual = x

        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.in_channels,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)
        x = nn.relu(x)
        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.in_channels,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)

        x = x + residual

        return nn.relu(x)
```
To do the same thing in PyTorch, we would write:
```python
class ResidualBlock(nn.Module):
    # One full block of a given filter size
    def __init__(self, in_filters):
        super(ResidualBlock, self).__init__()
        self.in_filters = in_filters
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                self.in_filters,
                self.in_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_filters),
            nn.ReLU(),
            nn.Conv2d(
                self.in_filters,
                self.in_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_filters),
        )

        #Requires _weights_init function
        self.apply(_weights_init)

    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        x += residual
        return F.relu(x)
```
While it felt awkward at the start, using Linen's API leads to shorter module definitions and easier-to-follow forward pass code.

## Train/Test Behaviour + State

ResNets employ Batch Normalization following convolutional layers. The BatchNorm layer is interesting as it:
- Has trainable parameters ($\alpha$ and $\beta$) and non-trainable variables (batch statistics)
- Has different train and test behaviour


Because of this, special care is required when implementing BatchNorm layers. First for the trainable and non-trainable parameters, we handle these in the model initialization. Calling the ```model.init(*)``` method returns a PyTree of all parameters. Since the BatchNorm parameters are the only non-trainable parameters, we can split them off as follows:

```python
...
variables = modelinit(rng, jnp.ones(input_shape))
params, batch_stats = variables["params"], variables["batch_stats"]
...
```

Managing train/eval behaviour is done by first adding a ```train``` bool to the ```__call__``` method of the main model (in this case the ResNet module), next we can initialize a partial module for a BatchNorm layer and pass it to all the necessary submodules. The following is a small section of the ResNet code:
```python
...
@nn.compact
def __call__(self, x, train):

    norm = partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.1,
        epsilon=1e-5,
        dtype=jnp.float32,
    )
    ...
    x = ResidualBlock(
                in_channels=16, norm=norm, dtype=jnp.float32
    )(x)
    ...
```
The final step is to add arugments to the model's ```.apply()``` method as follows:
```python
#Training 
logits, new_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch,
            mutable=["batch_stats"],
            train=True,
        )

#Evaluation - Use running mean of batch statistics
logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        batch,
        mutable=False,
        train=False,
    )
```

# Data Loading

In JAX/Flax, we can actually take the existing PyTorch data pipeline and modify it slightly to return jnp arrays instead of PyTorch Tensors. See [here](https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html) for more details. PyTorch's data loading and augmentations capabilities are great so being able to directly use this with Flax is great.

One issue I noticed was that my code would always return an OOM error if I set ```pin_memory=True``` in the dataloader. I suspect this is because JAX, by default will allocate tensors directly to the GPU memory, instead of the pinned memory. 

# Model Training

## TrainState
In Flax, all model training is passed through a ```TrainState``` class which holds the ```.apply()``` method, the optimizer, the model paramaters and any other attributes we wish to include. In the Flax example, I have created a subclass of ```TrainState``` and included batch statistics, weight decay, and dynamic scaling as extra attributes. 

## Optimizers
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

Adding a learning rate schedule is quite easy. Optax supports many of the common ones. Since the schdule is passed in as a function to the optimizer, all lr steps are handled internally compared with PyTorch which requires calling ```scheduler.step()``` manually. 

# Other Helpful resources
- Flax docs: [https://flax.readthedocs.io/en/latest/](https://flax.readthedocs.io/en/latest/)
- Flax examples: [https://github.com/google/flax/tree/main/examples](https://github.com/google/flax/tree/main/examples)
- EfficientNet + training script in Flax: [https://github.com/rwightman/efficientnet-jax](https://github.com/rwightman/efficientnet-jax)
- More ResNets in Flax: [https://github.com/n2cholas/jax-resnet](https://github.com/n2cholas/jax-resnet)
- Optax docs: [https://optax.readthedocs.io/en/latest/](https://optax.readthedocs.io/en/latest/)


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
