from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float32, PRNGKeyArray, jaxtyped

from .normalization import RMSLayerNorm

T = TypeVar("T")
Pair = tuple[T, T]


class PointWiseConv(eqx.Module):
    """Pointwise 1d convolution without bias."""

    weight: Array
    padding: tuple[int, int] | None = eqx.field(static=True)

    def __init__(
        self,
        size_in: int,
        size_out: int,
        padding: tuple[int, int] | None = None,
        *,
        key: PRNGKeyArray,
    ):
        self.weight = jnp.sqrt(2 / size_out) * jax.random.normal(
            key=key, shape=(size_out, size_in)
        )
        self.padding = padding

    def __call__(
        self, x: Float32[Array, " size_in time"]
    ) -> Float32[Array, " size_out _"]:
        y = jnp.dot(self.weight, x)
        if self.padding:
            return jnp.pad(y, [(0, 0), self.padding])
        return y


class PreGatedConv(eqx.Module):
    conv_1: PointWiseConv
    conv_2: PointWiseConv
    conv_gate: PointWiseConv
    gate_bias: Array
    total_bias: Array
    dilation: int = eqx.field(static=True)

    def __init__(
        self, size_in: int, size_out: int, dilation: int, *, key: PRNGKeyArray
    ):
        keys = jax.random.split(key, 3)
        self.conv_1 = PointWiseConv(size_in, size_out, (dilation, 0), key=keys[0])
        self.conv_2 = PointWiseConv(size_in, size_out, (dilation, 0), key=keys[1])
        self.conv_gate = PointWiseConv(size_in, size_out, (dilation, 0), key=keys[2])
        self.gate_bias = jnp.zeros((size_out, 1))
        self.total_bias = jnp.zeros((size_out, 1))
        self.dilation = dilation

    def __call__(
        self, x: Float32[Array, " size_in time"]
    ) -> Float32[Array, " size_out time"]:
        gate = jax.nn.sigmoid(self.conv_gate(x) + self.gate_bias)
        z_1 = self.conv_1(x)
        z_2 = self.conv_2(x)
        return jnp.tanh(
            z_1[:, self.dilation :] * gate[:, self.dilation :]
            + z_2[:, : -self.dilation] * gate[:, : -self.dilation]
            + self.total_bias
        )


class WavenetLayer(eqx.Module):
    pre_gated_conv: PreGatedConv
    residual_conv: PointWiseConv
    skip_conv: PointWiseConv
    norm: RMSLayerNorm

    def __init__(
        self,
        size_layers: int,
        dilation: int,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 3)
        self.pre_gated_conv = PreGatedConv(
            size_layers, size_layers, dilation, key=keys[0]
        )
        self.residual_conv = PointWiseConv(size_layers, size_layers, key=keys[1])
        self.skip_conv = PointWiseConv(size_layers, size_layers, key=keys[2])
        self.norm = RMSLayerNorm(size_layers)

    @jaxtyped
    @beartype
    def __call__(
        self,
        x: Float32[Array, " size_layers time"],
    ) -> Pair[Float32[Array, " size_layers time"]]:
        x_normalized = jax.vmap(self.norm, in_axes=1, out_axes=1)(x)
        z = self.pre_gated_conv(x_normalized)
        return self.residual_conv(z) + x, self.skip_conv(z)
