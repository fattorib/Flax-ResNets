import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import copy
from functools import partial
from utils_flax import compute_weight_decay
import numpy as np

ModuleDef = Any
dtypedef = Any


class ResidualBlock(nn.Module):
    # Define collection of datafields here
    in_channels: int

    # For batchnorm, you can pass it as a ModuleDef
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


class DownSampleResidualBlock(nn.Module):
    # Define collection of datafields here
    in_channels: int
    out_channels: int

    # For batchnorm, you can pass it as a ModuleDef
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
            strides=(2, 2),
            features=self.out_channels,
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)

        x = x + self.pad_identity(residual)

        return nn.relu(x)

    @nn.nowrap
    def pad_identity(self, x):
        # Pad identity connection when downsampling
        return jnp.pad(
            x[:, ::2, ::2, ::],
            ((0, 0), (0, 0), (0, 0), (self.out_channels // 4, self.out_channels // 4)),
            "constant",
        )


class ResNet(nn.Module):
    # Define collection of datafields here
    filter_list: Sequence[int]
    N: int
    num_classes: int

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv and linear layers
    kernel_init: Callable = nn.initializers.kaiming_normal()

    # For train/test differences, want to pass “mode switches” to __call__
    @nn.compact
    def __call__(self, x, train):

        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.1,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.filter_list[0],
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)

        x = norm()(x)
        x = nn.relu(x)

        # First stage
        for _ in range(0, self.N - 1):
            x = ResidualBlock(
                in_channels=self.filter_list[0], norm=norm, dtype=self.dtype
            )(x)

        x = DownSampleResidualBlock(
            in_channels=self.filter_list[0],
            out_channels=self.filter_list[1],
            norm=norm,
            dtype=self.dtype,
        )(x)

        # Second stage
        for _ in range(0, self.N - 1):
            x = ResidualBlock(
                in_channels=self.filter_list[1], norm=norm, dtype=self.dtype
            )(x)

        x = DownSampleResidualBlock(
            in_channels=self.filter_list[1],
            out_channels=self.filter_list[2],
            norm=norm,
            dtype=self.dtype,
        )(x)

        # Third stage
        for _ in range(0, self.N):
            x = ResidualBlock(
                in_channels=self.filter_list[2], norm=norm, dtype=self.dtype
            )(x)

        # Global pooling
        x = jnp.mean(x, axis=(1, 2))

        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(
            features=self.num_classes, kernel_init=self.kernel_init, dtype=self.dtype
        )(x)

        return x


def _resnet(layers, N, dtype=jnp.float32, num_classes=10):
    model = ResNet(filter_list=layers, N=N, dtype=dtype, num_classes=num_classes)
    return model


def ResNet20(
    dtype=jnp.float32,
):
    return _resnet(layers=[16, 32, 64], N=3, dtype=dtype, num_classes=10)


def ResNet32(
    dtype=jnp.float32,
):
    return _resnet(layers=[16, 32, 64], N=5, dtype=dtype, num_classes=10)


def ResNet44(
    dtype=jnp.float32,
):
    return _resnet(layers=[16, 32, 64], N=7, dtype=dtype, num_classes=10)


def ResNet56(
    dtype=jnp.float32,
):
    return _resnet(layers=[16, 32, 64], N=9, dtype=dtype, num_classes=10)


def ResNet110(
    dtype=jnp.float32,
):
    return _resnet(layers=[16, 32, 64], N=18, dtype=dtype, num_classes=10)
