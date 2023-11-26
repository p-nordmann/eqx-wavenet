import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float32, PRNGKeyArray, jaxtyped

from .normalization import RMSLayerNorm


class WavenetInput(eqx.Module):
    conv: eqx.nn.Conv1d
    norm: RMSLayerNorm

    def __init__(
        self,
        size_in: int,
        size_layers: int,
        input_kernel_size: int,
        *,
        key: PRNGKeyArray,
    ):
        self.conv = eqx.nn.Conv1d(
            in_channels=size_in,
            out_channels=size_layers,
            kernel_size=input_kernel_size,
            padding=[(input_kernel_size - 1, 0)],
            key=key,
        )
        self.norm = RMSLayerNorm(size_in)

    @jaxtyped
    @beartype
    def __call__(
        self, x: Float32[Array, " time size_in"]
    ) -> Float32[Array, " size_layers time"]:
        x_normalized = jax.vmap(self.norm)(x)
        return self.conv(jnp.transpose(x_normalized))
