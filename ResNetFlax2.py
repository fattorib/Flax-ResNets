import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import copy
from functools import partial
from utils_flax import compute_weight_decay

ModuleDef = Any

# ResNet implemenation without Sequential module

class ResidualBlock(nn.Module):
    # Define collection of datafields here
    in_channels: int
    out_channels: int
    
    # For batchnorm, you can pass it as a ModuleDef
    norm: ModuleDef

    # define init for conv layers
    kernel_init: Callable = nn.initializers.he_normal()

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
                )(x)
        x = self.norm()(x)
        nn.relu(x)
        nn.Conv(
                    kernel_size=(3, 3),
                    strides=1,
                    features=self.in_channels,
                    padding="SAME",
                    use_bias=False,
                    kernel_init=self.kernel_init,
                )(x)
        x = self.norm()(x)

        x += residual

        return nn.relu(x)

class DownSampleResidualBlock(nn.Module):
    # Define collection of datafields here
    in_channels: int
    out_channels: int
    
    # For batchnorm, you can pass it as a ModuleDef
    norm: ModuleDef

    # downsample_func: Any

    # define init for conv layers
    kernel_init: Callable = nn.initializers.he_normal()

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
                    )(x)
        x = self.norm()(x)
        x = nn.Conv(
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        features=self.out_channels,
                        padding="SAME",
                        use_bias=False,
                        kernel_init=self.kernel_init,
            )(x)

        x += self.pad_identity(residual)

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

    # define init for conv layers
    kernel_init: Callable = nn.initializers.he_normal()

    # For train/test differences, want to pass “mode switches” to __call__
    @nn.compact
    def __call__(self, x, train):

        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
        )

        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.filter_list[0],
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
        )(x)


        x = norm()(x)
        x = nn.relu(x)

        #First stage
        for i in range(0,2*self.N-1):
            x = ResidualBlock(in_channels=self.filter_list[0],
            out_channels=self.filter_list[0],
            norm=norm
            )(x)
        
        x = DownSampleResidualBlock(in_channels=self.filter_list[0],
            out_channels=self.filter_list[1],
            norm=norm)(x)

        #Second stage
        for i in range(0,2*self.N-1):
            x = ResidualBlock(in_channels=self.filter_list[1],
            out_channels=self.filter_list[1],
            norm=norm
            )(x)
        
        x = DownSampleResidualBlock(in_channels=self.filter_list[1],
            out_channels=self.filter_list[2],
            norm=norm)(x)

        #Third stage
        for i in range(0,2*self.N):
            x = ResidualBlock(in_channels=self.filter_list[2],
            out_channels=self.filter_list[2],
            norm=norm
            )(x)
        
        # Global pooling
        x = jnp.mean(x, axis=(1, 2))

        x = x.reshape(x.shape[0], -1)

        x = nn.Dense(features=self.num_classes, kernel_init=self.kernel_init)(x)

        return x


def _resnet(layers, N, num_classes=10):
    model = ResNet(filter_list=layers, N=N, num_classes=num_classes)
    return model


def ResNet20():
    return _resnet(layers=[16, 32, 64], N=3, num_classes=10)


def ResNet32():
    return _resnet(layers=[16, 32, 64], N=5, num_classes=10)

def ResNet44():
    return _resnet(layers=[16, 32, 64], N=7, num_classes=10)


def ResNet56():
    return _resnet(layers=[16, 32, 64], N=9, num_classes=10)


def ResNet110():
    return _resnet(layers=[16, 32, 64], N=18, num_classes=10)


if __name__ == "__main__":

    model = ResNet20()

    rng = jax.random.PRNGKey(0)

    params = model.init(rng, jnp.ones([1, 32, 32, 3]), train=True)["params"]
    batch_stats = model.init(rng, jnp.ones([1, 32, 32, 3]), train=True)["batch_stats"]

    test_batch = jnp.ones([128, 32, 32, 3])

    batch_out, state, = model.apply(
        {"params": params, "batch_stats": batch_stats},
        test_batch,
        train=True,
        mutable=["batch_stats"],
    )
