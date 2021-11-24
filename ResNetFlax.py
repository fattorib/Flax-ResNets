import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import copy
from functools import partial

ModuleDef = Any

# Custom Sequential does not like nn.relu(), it is not a flax module
# One option is to just make a relu module
class Sequential(nn.Module):
    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        for layer in self.layers:
            x = layer(x)
        return x


class RelU(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        return nn.relu(inputs)


class ResidualBlock(nn.Module):
    # Define collection of datafields here
    in_channels: int
    out_channels: int
    N: int

    # For batchnorm, you can pass it as a ModuleDef
    norm: ModuleDef

    downsample: bool = True

    # define init for conv layers
    kernel_init: Callable = nn.initializers.he_normal()

    def setup(self):
        # setup function is called at the end of __postinit__
        # allows you to name sublayers
        # required if using multiple class methods (as we will be)

        layer = Sequential(
            [
                nn.Conv(
                    kernel_size=(3, 3),
                    strides=1,
                    features=self.in_channels,
                    padding="SAME",
                    use_bias=False,
                    kernel_init=self.kernel_init,
                ),
                self.norm(),
                RelU(),
                nn.Conv(
                    kernel_size=(3, 3),
                    strides=1,
                    features=self.in_channels,
                    padding="SAME",
                    use_bias=False,
                    kernel_init=self.kernel_init,
                ),
                self.norm(),
            ]
        )

        self.layers = [copy.deepcopy(layer) for _ in range(self.N - 1)]

        if self.downsample:
            self.finallayer = Sequential(
                [
                    nn.Conv(
                        kernel_size=(3, 3),
                        strides=1,
                        features=self.in_channels,
                        padding="SAME",
                        use_bias=False,
                        kernel_init=self.kernel_init,
                    ),
                    self.norm(),
                    RelU(),
                    nn.Conv(
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        features=self.out_channels,
                        padding="SAME",
                        use_bias=False,
                        kernel_init=self.kernel_init,
                    ),
                    self.norm(),
                ]
            )
        else:
            self.finallayer = Sequential(
                [
                    nn.Conv(
                        kernel_size=(3, 3),
                        strides=1,
                        features=self.in_channels,
                        padding="SAME",
                        use_bias=False,
                        kernel_init=self.kernel_init,
                    ),
                    self.norm(),
                    RelU(),
                    nn.Conv(
                        kernel_size=(3, 3),
                        strides=1,
                        features=self.out_channels,
                        padding="SAME",
                        use_bias=False,
                        kernel_init=self.kernel_init,
                    ),
                    self.norm(),
                ]
            )

    def __call__(self, input):
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

    # @jax.jit
    def pad_identity(self, x):
        # Pad identity connection in the case of downsampling
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

        x = ResidualBlock(
            in_channels=self.filter_list[0],
            out_channels=self.filter_list[1],
            N=self.N,
            norm=norm,
        )(x)

        x = ResidualBlock(
            in_channels=self.filter_list[1],
            out_channels=self.filter_list[2],
            N=self.N,
            norm=norm,
        )(x)

        x = ResidualBlock(
            in_channels=self.filter_list[2],
            out_channels=self.filter_list[2],
            N=self.N,
            downsample=False,
            norm=norm,
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


def ResNet50():
    return _resnet(layers=[16, 32, 64], N=8, num_classes=10)


def ResNet110():
    return _resnet(layers=[16, 32, 64], N=18, num_classes=10)


if __name__ == "__main__":

    model = ResNet20()

    rng = jax.random.PRNGKey(0)

    params = model.init(rng, jnp.ones([1, 32, 32, 3]), train=True)["params"]
    batch_stats = model.init(rng, jnp.ones([1, 32, 32, 3]), train=True)["batch_stats"]

    test_batch = jnp.ones([128, 32, 32, 3])

    batch_out, state = model.apply(
        {"params": params, "batch_stats": batch_stats},
        test_batch,
        train=True,
        mutable=["batch_stats"],
    )

    print(batch_out.shape)
