import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from eqx_wavenet import Wavenet, WavenetConfig


def compute_loss(model, inputs):
    preds = jax.vmap(model)(inputs)[:, :-1]
    return jnp.mean(optax.l2_loss(preds, inputs[:, 1:]))


@eqx.filter_jit
def make_step(model, inputs, opt, opt_state):
    _, grads = eqx.filter_value_and_grad(compute_loss)(model, inputs)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state


@eqx.filter_jit
def make_eval_step(model, inputs):
    # For evaluaation, we will use L1 loss instead.
    preds = jax.vmap(model)(inputs)[:, :-1]
    return jnp.mean(jnp.absolute(preds - inputs[:, 1:]))


def make_epoch(data, window_size, batch_size, *, key):
    n = data.shape[0]

    # Make permutation of the data.
    key, key_permutation = jax.random.split(key)
    idx = jax.random.permutation(x=data.shape[0], key=key_permutation)

    # Build batches.
    for k in range(0, n - window_size, batch_size):
        if k + batch_size > n - window_size:  # Skip the end
            break
        batch = jnp.zeros((batch_size, window_size), dtype=float)
        for j in range(window_size):
            batch = batch.at[:, j].set(data[idx[k : k + batch_size] + j])
        yield batch[:, :, jnp.newaxis]


def test_training_sinusoid():
    """Training on a sinusoid to make sure it learns something."""

    # Constants
    seed = 0
    n = 10_000
    n_train = 8_000
    sine_period = 100
    noise_std = 0.3
    wavenet_config = WavenetConfig(
        num_layers=4,
        layer_dilations=[1, 2, 4, 8],
        size_in=1,
        input_kernel_size=1,
        size_layers=16,
        size_hidden=64,
        size_out=1,
    )
    learning_rate = 1e-3
    batch_size = 50
    window_size = 30
    expected_improvement_factor = (
        2  # Completely arbitrary, but loss should significatively improve
    )

    # Make PRNGKey.
    key = jax.random.PRNGKey(seed)

    # Make sinusoidal data.
    key, key_data = jax.random.split(key)
    data = (
        jnp.sin(jnp.arange(n) * (2 * jnp.pi) / sine_period)
        + jax.random.normal(shape=(n,), key=key_data) * noise_std
    )
    data_train, data_test = data[:n_train], data[n_train:]

    # Make model.
    key, key_model = jax.random.split(key)
    model = Wavenet(
        config=wavenet_config,
        key=key_model,
    )

    # Make optimizer.
    opt = optax.adam(learning_rate)
    opt_state = opt.init(model)

    # Eval model before training.
    key, key_epoch = jax.random.split(key)
    losses_before = []
    for inputs in make_epoch(
        data=data_test, window_size=window_size, batch_size=batch_size, key=key_epoch
    ):
        loss = make_eval_step(model, inputs)
        losses_before.append(loss)

    # Train model after training.
    key, key_epoch = jax.random.split(key)
    for inputs in make_epoch(
        data=data_train, window_size=window_size, batch_size=batch_size, key=key_epoch
    ):
        model, opt_state = make_step(model, inputs, opt, opt_state)

    # Eval model.
    key, key_epoch = jax.random.split(key)
    losses_after = []
    for inputs in make_epoch(
        data=data_test, window_size=window_size, batch_size=batch_size, key=key_epoch
    ):
        loss = make_eval_step(model, inputs)
        losses_after.append(loss)

    # Check that loss is good.
    before, after = jnp.mean(jnp.stack(losses_before)), jnp.mean(
        jnp.stack(losses_after)
    )
    assert before / expected_improvement_factor > after
