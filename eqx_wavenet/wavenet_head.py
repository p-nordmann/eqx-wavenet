import equinox as eqx
import jax
from beartype import beartype
from jaxtyping import Array, Float32, PRNGKeyArray, jaxtyped

from .normalization import RMSLayerNorm


class WavenetHead(eqx.Module):
    linear_in: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    norm: RMSLayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        size_layers: int,
        size_hidden: int,
        size_out: int,
        *,
        key: PRNGKeyArray,
    ):
        key_1, key_2 = jax.random.split(key)
        self.linear_in = eqx.nn.Linear(size_layers, size_hidden, key=key_1)
        self.linear_out = eqx.nn.Linear(size_hidden, size_out, key=key_2)
        self.norm = RMSLayerNorm(size_layers)
        self.dropout = eqx.nn.Dropout()

    @jaxtyped
    @beartype
    def __call__(
        self,
        x: Float32[Array, " size_layers"],
        enable_dropout: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> Float32[Array, " size_out"]:
        x_normalized = self.norm(x)
        z = jax.nn.relu(self.linear_in(x_normalized))
        z = self.dropout(z, inference=not enable_dropout, key=key)
        return self.linear_out(z)
