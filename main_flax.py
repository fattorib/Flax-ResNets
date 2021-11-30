import argparse
import shutil
import time

from typing import Callable, Any
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
from torch.utils import data
from utils_flax import (
    NumpyLoader,
    FlattenAndCast,
    create_cos_anneal_schedule,
    compute_weight_decay,
)
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from ResNetFlax import ResNet20, ResNet32, ResNet44, ResNet56, ResNet110
import flax.linen as nn
import optax
from flax.training import train_state
import numpy as np
import torch.utils
import wandb


class TrainState(train_state.TrainState):
    batch_stats: Any = None
    weight_decay: Any = None


def parse():
    parser = argparse.ArgumentParser(description="Flax CIFAR10 Training")

    parser.add_argument(
        "-data",
        "--data",
        default="ML/",
        type=str,
        metavar="DIR",
        help="path to dataset",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )

    parser.add_argument(
        "--epochs",
        default=120,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size per process (default: 128)",
    )

    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )

    parser.add_argument(
        "--print-freq",
        "-p",
        default=100,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )

    parser.add_argument("--local_rank", default=0, type=int)

    # My additional args
    parser.add_argument("--model", type=str, default="ResNet20")
    parser.add_argument("--CIFAR10", type=bool, default=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--cos-anneal", type=bool, default=False)
    parser.add_argument("--base-lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)

    args = parser.parse_args()
    return args


def main():
    global best_prec1, args

    args = parse()

    if args.model == "ResNet20":
        model = ResNet20()

    elif args.model == "ResNet32":
        model = ResNet32()

    elif args.model == "ResNet44":
        model = ResNet44()

    elif args.model == "ResNet56":
        model = ResNet56()

    elif args.model == "ResNet110":
        model = ResNet110()

    if args.CIFAR10:
        assert args.num_classes == 10, "Must have 10 output classes for CIFAR10"
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
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
                FlattenAndCast(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
                FlattenAndCast(),
            ]
        )

        train_dataset = CIFAR10(
            root="./CIFAR", train=True, download=False, transform=transform_train
        )
        train_dataset, validation_dataset = torch.utils.data.random_split(
            train_dataset, [45000, 5000]
        )

        test_dataset = CIFAR10(
            root="./CIFAR", train=False, download=False, transform=transform_test
        )

        train_loader = NumpyLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False,
        )

        validation_loader = NumpyLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False,
        )

        test_loader = NumpyLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
        )

    # Setup WandB logging here
    wandb_run = wandb.init(project="Flax Torch")
    wandb.config.max_epochs = args.epochs
    wandb.config.batch_size = args.batch_size
    wandb.config.weight_decay = args.weight_decay

    wandb.config.ModelName = args.model
    wandb.config.Dataset = "CIFAR10"
    wandb.config.Package = "Flax"

    # --------- Create Train State ---------#
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    learning_rate_fn = optax.piecewise_constant_schedule(
        init_value=args.base_lr, boundaries_and_scales={32000: 0.1, 48000: 0.1}
    )

    state = create_train_state(
        init_rng,
        momentum=args.momentum,
        learning_rate_fn=learning_rate_fn,
        weight_decay=args.weight_decay,
        model=model,
    )
    del init_rng

    for epoch in range(0, args.epochs):
        state, train_epoch_metrics_np = train_epoch(state, train_loader, epoch)

        print(
            f"train epoch: {epoch}, loss: {train_epoch_metrics_np['loss']:.4f}, accuracy:{train_epoch_metrics_np['accuracy']*100:.2f}%"
        )

        # Get LR:
        lr = learning_rate_fn(epoch * args.batch_size)
        lr_np = jax.device_get(lr)
        # lr_np = args.base_lr

        # Validation set metrics:
        validation_loss, _ = eval_model(state, validation_loader)

        if epoch % 10 == 0:
            _, test_accuracy = eval_model(state, test_loader)

            wandb.log(
                {
                    "acc@1": test_accuracy * 100,
                    "Learning Rate": lr_np,
                    "Training Loss": train_epoch_metrics_np["loss"],
                    "Validation Loss": validation_loss,
                }
            )

        else:
            wandb.log(
                {
                    "Learning Rate": lr_np,
                    "Training Loss": train_epoch_metrics_np["loss"],
                    "Validation Loss": validation_loss,
                }
            )


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


def create_train_state(rng, momentum, learning_rate_fn, weight_decay, model):
    """Creates initial `TrainState`."""
    params, batch_stats = initialized(rng, 32, model)
    tx = optax.sgd(learning_rate=learning_rate_fn, momentum=momentum, nesterov=True)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        weight_decay=weight_decay,
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

        loss = loss + 0.5 * state.weight_decay * compute_weight_decay(params)

        return loss, (logits, new_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    logits, new_state = aux[1]

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
    return compute_metrics(logits=logits, labels=labels)


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
    main()
