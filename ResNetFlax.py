import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import copy

class Sequential(nn.Module):
    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self,inputs):
        x = inputs

        for layer in self.layers:
            x = layer(x)
        return x 




class ResidualBlock(nn.Module):
    # Define collection of datafields here
    in_channels: int
    out_channels: int
    N: int
    downsample: bool

    #define init for conv layers
    kernel_init: Callable = nn.initializers.he_normal()

    def setup(self):
        # setup function is called at the end of __postinit__
        # allows you to name sublayers
        # required if using multiple class methods (as we will be)

        layer = Sequential([
            nn.Conv(
                kernel_size = 3,
                strides = 1,
                padding = 'SAME',
                use_bias = False,
                kernel_init=self.kernel_init
                ),
                #BatchNorm
                nn.relu(),
            nn.Conv(
                kernel_size = 3,
                strides = 1,
                padding = 'SAME',
                use_bias = False,
                kernel_init=self.kernel_init
                ),
                #BatchNorm
        ])


        self.layers = [copy.deepcopy(layer) for _ in range(self.N - 1)]

        if self.downsample:
            self.finallayer = Sequential([
            nn.Conv(
                kernel_size = 3,
                strides = 1,
                padding = 'SAME',
                use_bias = False,
                kernel_init=self.kernel_init
                ),
                #BatchNorm
                nn.relu(),
            nn.Conv(
                kernel_size = 3,
                strides = (2,2),
                padding = 'SAME',
                use_bias = False,
                kernel_init=self.kernel_init
                ),
                #BatchNorm
            ])
        else:
            self.finallayer = Sequential([
            nn.Conv(
                kernel_size = 3,
                strides = 1,
                padding = 'SAME',
                use_bias = False,
                kernel_init=self.kernel_init
                ),
                #BatchNorm
                nn.relu(),
            nn.Conv(
                kernel_size = 3,
                strides = 1,
                padding = 'SAME',
                use_bias = False,
                kernel_init=self.kernel_init
                ),
                #BatchNorm
            ])


    def __call__(self, input, train = True):
        x = input

        for layer in self.layers:
            residual = x
            x = layer(x)
            x += residual

            x = nn.relu(x)
        
        residual = x

        x = self.finallayer(x)

        if self.downsample:
            x += self.pad_identity(residual)

        else:
            x += residual

        return nn.relu(x)


    def pad_identity(self, x):
        # Pad identity connection in the case of downsampling
        return jnp.pad(
            x[:, :, ::2, ::2],
            (0, 0, 0, 0, self.out_channels // 4, self.out_channels // 4),
            "constant",
            0,
        )
