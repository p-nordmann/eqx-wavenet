from typing import TypeVar

import equinox as eqx
import jax
from beartype import beartype
from jaxtyping import Array, Float32, PRNGKeyArray, jaxtyped

from .normalization import RMSLayerNorm

T = TypeVar("T")
Pair = tuple[T, T]


def dilated_conv(
    size_layers: int,
    dilation: int,
    *,
    key: PRNGKeyArray,
) -> eqx.nn.Conv1d:
    return eqx.nn.Conv1d(
        in_channels=size_layers,
        out_channels=size_layers,
        kernel_size=2,
        dilation=dilation,
        padding=[(dilation, 0)],  # We try to preserve the same length as input.
        key=key,
    )


def pointwise_conv(
    size_layers: int,
    *,
    key: PRNGKeyArray,
) -> eqx.nn.Conv1d:
    return eqx.nn.Conv1d(
        in_channels=size_layers,
        out_channels=size_layers,
        kernel_size=1,
        key=key,
    )


class WavenetLayer(eqx.Module):
    filter_conv: eqx.nn.Conv1d
    gate_conv: eqx.nn.Conv1d
    residual_conv: eqx.nn.Conv1d
    skip_conv: eqx.nn.Conv1d
    norm: RMSLayerNorm

    def __init__(
        self,
        size_layers: int,
        dilation: int,
        *,
        key: PRNGKeyArray,
    ):
        key_1, key_2, key_3, key_4 = jax.random.split(key, 4)

        self.filter_conv = dilated_conv(size_layers, dilation, key=key_1)
        self.gate_conv = dilated_conv(size_layers, dilation, key=key_2)

        self.residual_conv = pointwise_conv(size_layers, key=key_3)
        self.skip_conv = pointwise_conv(size_layers, key=key_4)

        self.norm = RMSLayerNorm(size_layers)

    @jaxtyped
    @beartype
    def __call__(
        self,
        x: Float32[Array, " size_layers time"],
    ) -> Pair[Float32[Array, " size_layers time"]]:
        x_normalized = jax.vmap(self.norm, in_axes=1, out_axes=1)(x)
        z = jax.nn.tanh(self.filter_conv(x_normalized)) * jax.nn.sigmoid(
            self.gate_conv(x_normalized)
        )
        return self.residual_conv(z) + x, self.skip_conv(z)
