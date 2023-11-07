import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float32, PRNGKeyArray, jaxtyped

from .wavenet_config import WavenetConfig
from .wavenet_head import WavenetHead
from .wavenet_input import WavenetInput
from .wavenet_layer import WavenetLayer


class Wavenet(eqx.Module):
    wavenet_input: WavenetInput
    wavenet_layers: list[WavenetLayer]
    wavenet_head: WavenetHead

    def __init__(
        self,
        config: WavenetConfig,
        *,
        key: PRNGKeyArray,
    ):
        key_input, key = jax.random.split(key)
        self.wavenet_input = WavenetInput(
            config.size_in,
            config.size_layers,
            config.input_kernel_size,
            key=key_input,
        )

        assert config.num_layers == len(config.layer_dilations)
        self.wavenet_layers = []
        for k in range(config.num_layers):
            key_layer, key = jax.random.split(key)
            self.wavenet_layers.append(
                WavenetLayer(
                    config.size_layers,
                    config.layer_dilations[k],
                    key=key_layer,
                )
            )

        key_head, key = jax.random.split(key)
        self.wavenet_head = WavenetHead(
            config.size_layers,
            config.size_hidden,
            config.size_out,
            key=key_head,
        )

    @jaxtyped
    @beartype
    def __call__(
        self,
        x: Float32[Array, " time size_in"],
        *,
        enable_dropout: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> Float32[Array, " time size_out"]:
        z = self.wavenet_input(x)
        sum_out = jnp.zeros_like(z)
        for layer in self.wavenet_layers:
            z, out = layer(z)
            sum_out += out
        out = jax.nn.relu(sum_out)
        return jax.vmap(
            self.wavenet_head,
            in_axes=[1, None, None],
        )(
            out,
            enable_dropout,
            key,
        )
