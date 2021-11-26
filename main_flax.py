from ResNetFlax import ResNet
from typing import Callable, Any
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
from torch.utils import data
from utils_flax import NumpyLoader, FlattenAndCast, create_cos_anneal_schedule
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import flax.linen as nn
import optax
from flax.training import train_state
import numpy as np


class TrainState(train_state.TrainState):
    batch_stats: Any = None


@jax.jit
def cross_entropy_loss(*, logits, labels):
    """
    Softmax + CE Loss
    """
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * nn.log_softmax(logits, axis=-1), axis=-1))


def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(rng, shape):
        return model.init(rng, shape, train=True)

    variables = init(rng=key, shape=jnp.ones(input_shape))
    return variables["params"], variables["batch_stats"]


def create_train_state(rng, momentum, learning_rate_fn):
    """Creates initial `TrainState`."""
    model = ResNet(filter_list=[16, 32, 64], N=3, num_classes=10)
    params, batch_stats = initialized(rng, 32, model)
    tx = optax.sgd(learning_rate=learning_rate_fn, momentum=momentum, nesterov=True)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )
    return state


@jax.jit
def train_step(state, batch, labels):
    """Train for a single step."""

    def loss_fn(params):
        logits, new_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch,
            mutable=["batch_stats"],
            train=True,
        )
        loss = cross_entropy_loss(logits=logits, labels=labels)
        return loss, (logits, new_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    logits, new_state = aux[1]

    # #Maybe for logging - want to do this by epoch, not step
    # step = state.step
    # lr = learning_rate_fn(step)

    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_state["batch_stats"],
    )
    metrics = compute_metrics(logits=logits, labels=labels)

    return state, metrics


@jax.jit
def eval_step(state, batch, labels):
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        batch,
        mutable=False,
        train=False,
    )
    return compute_metrics(logits=logits, targets=labels)


def train_epoch(state, dataloader, epoch):
    """Train for a single epoch."""
    batch_metrics = []

    for images, labels in dataloader:
        state, metrics = train_step(state, images, labels)
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    return state, epoch_metrics_np


def eval_model(state, dataloader):
    batch_metrics = []
    for images, labels in dataloader:
        metrics = eval_step(state, images, labels)
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(batch_metrics)
    validation_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    return validation_metrics_np["loss"], validation_metrics_np["accuracy"]


if __name__ == "__main__":

    # PyTorch dataloading
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(
                (32, 32),
                padding=4,
                fill=0,
                padding_mode="constant",
            ),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            FlattenAndCast(),
        ]
    )

    train_dataset = CIFAR10(
        root="./CIFAR", train=True, download=False, transform=transform_train
    )

    trainloader = NumpyLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )

    num_epochs = 10

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    learning_rate = 0.1
    momentum = 0.9

    learning_rate_fn = create_cos_anneal_schedule(
        base_lr=0.1, min_lr=0.001, max_steps=500
    )

    state = create_train_state(init_rng, momentum, learning_rate_fn=learning_rate_fn)
    del init_rng  # Must not be used anymore.

    for epoch in range(1, num_epochs + 1):
        # Run an optimization step over a training batch
        state, epoch_metrics_np = train_epoch(state, trainloader, epoch)

        print(
            f"train epoch: {epoch}, loss: {epoch_metrics_np['loss']:.4f}, accuracy:{epoch_metrics_np['accuracy']*100:.2f}%"
        )

        # Evaluate on validation set

        # val_loss, val_acc = eval_model(state, dataloader=valloader)

        # Evaluate on test set
        # test_loss, test_acc = eval_model(state, dataloader=testloader)
