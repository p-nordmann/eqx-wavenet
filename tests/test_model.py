import jax
import jax.numpy as jnp

from eqx_wavenet.normalization import RMSLayerNorm


def test_rms_layernorm():
    eps = 1e-2
    norm = RMSLayerNorm(10)
    x = jnp.reshape(jnp.arange(0, 20, dtype=jnp.float32), newshape=(2, 10))
    got = jax.vmap(norm)(x)
    want = x / jnp.array([[5.339] * 10, [14.782] * 10], dtype=jnp.float32)
    assert jnp.allclose(got, want, atol=eps).item()
