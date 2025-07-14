import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Sequence, Dict, Any
from tools.plot_utils import *
from jax_models.neural_odes import NeuralODE
import optax
import time
import equinox as eqx
import os


@dataclass
class Config:
    """Configuration class for Neural ODE training."""
    # Dataset parameters
    dataset_size: int = 100
    time_grid_start: float = 0.0
    time_grid_end: float = 5.0
    time_grid_points: int = 200

    # Training parameters
    batch_size: int = 100
    learning_rate: float = 5e-4
    steps_strategy: Tuple[int, ...] = (3000, 6000)
    length_strategy: Tuple[float, ...] = (0.5, 1.0)

    # Model parameters
    width_size: int = 200
    depth: int = 1
    lambda_reg: float = 1e-6

    # Optimization parameters
    decay_rate: float = 0.9
    transition_steps: int = 1000

    # General parameters
    seed: int = 8
    print_every: int = 100

    # Plotting parameters
    plot: bool = True
    plot_samples: int = 100
    plot_max_trajectories: int = 20
    plot_save_path: str = "figures/jax_Compare.png"
    plot_dpi: int = 300
    plot_figsize: Tuple[int, int] = (16, 6)

    # Animation parameters
    anim: bool = True
    animation_save_path: str = "figures/detailed_animation.gif"
    fps: int = 15
    interval: int = 80
    trajectory_idx: int = 10

    # Model saving/loading
    model_save_path: str = "models/neural_ode_model.pkl"
    load_existing: bool = False
    save_final: bool = True
    save_intermediate: bool = True

    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """Load configuration from JSON file."""
        print(f"Loading configuration from {json_path}...")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Configuration file not found: {json_path}")

        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        # Convert lists back to tuples for strategies
        if 'steps_strategy' in config_dict:
            config_dict['steps_strategy'] = tuple(config_dict['steps_strategy'])
        if 'length_strategy' in config_dict:
            config_dict['length_strategy'] = tuple(config_dict['length_strategy'])
        if 'plot_figsize' in config_dict:
            config_dict['plot_figsize'] = tuple(config_dict['plot_figsize'])

        print("✅ Configuration loaded successfully!")
        return cls(**config_dict)

    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file."""
        print(f"Saving configuration to {json_path}...")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        # Convert dataclass to dict
        config_dict = asdict(self)

        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print("✅ Configuration saved successfully!")

    def print_config(self) -> None:
        """Print current configuration in a formatted way."""
        print("Configuration:")
        print(f"   Dataset:")
        print(f"     - Size: {self.dataset_size}")
        print(f"     - Time grid: {self.time_grid_start} to {self.time_grid_end} ({self.time_grid_points} points)")
        print(f"   Training:")
        print(f"     - Batch size: {self.batch_size}")
        print(f"     - Learning rate: {self.learning_rate}")
        print(f"     - Steps strategy: {self.steps_strategy}")
        print(f"     - Length strategy: {self.length_strategy}")
        print(f"   Model:")
        print(f"     - Width: {self.width_size}, Depth: {self.depth}")
        print(f"     - Regularization: {self.lambda_reg}")
        print(f"   General:")
        print(f"     - Seed: {self.seed}")
        print(f"     - Model path: {self.model_save_path}")
        print(f"     - Load existing: {self.load_existing}")
        print(f"     - Save final: {self.save_final}")
        print(f"   Plotting:")
        print(f"     - Enabled: {self.plot}")
        print(f"     - Enabled: {self.anim}")


def create_default_configs():
    """Create some default configuration files for common scenarios."""
    configs_dir = "configs"
    os.makedirs(configs_dir, exist_ok=True)

    # Default training config
    default_config = Config()
    default_config.to_json(f"{configs_dir}/default.json")

    # Fast training config for testing
    fast_config = Config(
        dataset_size=50,
        batch_size=50,
        steps_strategy=(100, 500),
        length_strategy=(0.5, 1.0),
        width_size=50,
        depth=1,
        plot_samples=20
    )
    fast_config.to_json(f"{configs_dir}/fast.json")

    # High quality config
    high_quality_config = Config(
        dataset_size=500,
        batch_size=100,
        steps_strategy=(2000, 5000),
        length_strategy=(0.5, 1.0),
        width_size=300,
        depth=3,
        learning_rate=5e-4,
        plot_samples=200
    )
    high_quality_config.to_json(f"{configs_dir}/high_quality.json")

    # Inference only config
    inference_config = Config(
        load_existing=True,
        steps_strategy=(),
        plot=True,
        save_final=False,
        save_intermediate=False
    )
    inference_config.to_json(f"{configs_dir}/inference.json")

    print("✅ Default configuration files created in 'configs/' directory")


def save_model(model, filepath):
    """Save a JAX/Equinox model to disk."""
    print(f"Saving model to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Use equinox's tree_serialise_leaves for JAX compatibility
    with open(filepath, 'wb') as f:
        eqx.tree_serialise_leaves(f, model)
    print(f"✅ Model saved successfully!")


def load_model(filepath, model_template):
    """Load a JAX/Equinox model from disk."""
    print(f"Loading model from {filepath}...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Use equinox's tree_deserialise_leaves for JAX compatibility
    with open(filepath, 'rb') as f:
        model = eqx.tree_deserialise_leaves(f, model_template)

    print(f"✅ Model loaded successfully!")
    return model


def model_exists(filepath):
    """Check if a model file exists."""
    return os.path.exists(filepath)


# Ground truth ODE
def data_equation(t, y, params):
    """
    Modified oscillator dynamics with hyperbolic secant nonlinearity
    y = [position, velocity] or [x, y] coordinates
    """
    dydt = jnp.array([
        y[1],  # dx/dt
        (1-y[0]**2)*y[1] - y[0]  # dy/dt
    ])
    return dydt


# Dataset generation
def _get_data(time_grid, *, key):
    y0 = jr.uniform(key, (2,), minval=-3, maxval=3)
    params = 1  # Not needed in this example

    def f(t, y, _):  # args unused
        return data_equation(t, y, params)

    solver = diffrax.Dopri5()
    dt0 = 0.01
    saveat = diffrax.SaveAt(ts=time_grid)
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(f),
        solver,
        t0=time_grid[0],
        t1=time_grid[-1],
        dt0=dt0,
        y0=y0,
        saveat=saveat
    )
    return solution.ys


def get_data(config: Config, *, key):
    """Generate dataset using configuration parameters."""
    print(f"Generating dataset with {config.dataset_size} samples...")
    time_grid = jnp.linspace(config.time_grid_start, config.time_grid_end, config.time_grid_points)
    print(f"Time grid created with {len(time_grid)} points from {time_grid[0]:.1f} to {time_grid[-1]:.1f}")

    subkeys = jr.split(key, config.dataset_size)
    print(f"Split random keys for {len(subkeys)} trajectories")

    print("Computing trajectories (this might take a moment)...")
    trajectories = jax.vmap(lambda k: _get_data(time_grid, key=k))(subkeys)
    print(f"✅ Dataset generated! Shape: {trajectories.shape}")

    return time_grid, trajectories


# Random batch iterator
def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(arr.shape[0] == dataset_size for arr in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        permutation = jr.permutation(key, indices)
        key, = jr.split(key, 1)
        for start in range(0, dataset_size - batch_size + 1, batch_size):
            batch_indices = permutation[start:start + batch_size]
            yield tuple(arr[batch_indices] for arr in arrays)


def main(config: Optional[Config] = None, config_path: Optional[str] = None):
    """
    Main training function using configuration.

    Args:
        config: Config object (takes precedence over config_path)
        config_path: Path to JSON configuration file
    """
    if config is None:
        if config_path is not None:
            config = Config.from_json(config_path)
        else:
            config = Config()

    print("\n🚀 Starting Neural ODE training...")
    config.print_config()

    # Seed and split RNG
    key = jr.PRNGKey(config.seed)
    data_key, _, model_key, loader_key = jr.split(key, 4)

    # Data
    print("\nGenerating training data...")
    ts, ys = get_data(config, key=data_key)
    _, traj_len, dim = ys.shape
    print(f"✅ Data ready! Trajectories shape: {ys.shape} (samples × time × dim)")

    # Model initialization or loading
    print(f"\nSetting up Neural ODE model...")

    if config.load_existing and model_exists(config.model_save_path):
        # Create template model for loading
        template_model = NeuralODE(dim, config.width_size, config.depth, key=model_key)
        try:
            model = load_model(config.model_save_path, template_model)
            print("Using loaded model for training/inference")
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            print("Creating new model...")
            model = NeuralODE(dim, config.width_size, config.depth, key=model_key)
    else:
        if config.load_existing:
            print(f"Model file not found at {config.model_save_path}")
        print("Creating new model...")
        model = NeuralODE(dim, config.width_size, config.depth, key=model_key)

    print("✅ Model ready")

    @eqx.filter_value_and_grad
    @eqx.filter_jit
    def loss_fn(model, t, y_true):
        """Loss function with L2 regularization."""
        # Prediction from the model
        y_pred = jax.vmap(model, in_axes=(None, 0))(t, y_true[:, 0])

        # Mean squared error loss
        mse_loss = jnp.mean((y_true - y_pred) ** 2)

        def add_weight_norm(leaf):
            if hasattr(leaf, 'weight'):
                return jnp.sum(leaf.weight ** 2)
            return 0.0

        # Sum L2 norms across all model parameters
        l2_loss = sum(jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(add_weight_norm, model)
        ))

        # Total loss (MSE + regularization)
        total_loss = mse_loss + config.lambda_reg * l2_loss
        return total_loss

    @eqx.filter_jit
    def training_step(t, y_batch, model, opt_state):
        loss, grads = loss_fn(model, t, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    t_global_start = time.time()

    # Training loop
    for phase_idx, (n_steps, frac_len) in enumerate(zip(config.steps_strategy, config.length_strategy)):
        print(f"\n=== TRAINING PHASE {phase_idx + 1}/{len(config.steps_strategy)} ===")

        schedule = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.transition_steps,
            decay_rate=config.decay_rate,
            staircase=True
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        # Train on equispaced points of the trajectory
        n_points = int(traj_len * frac_len)
        idx = jnp.linspace(0, traj_len - 1, n_points).round().astype(int)

        ts_train = ts[idx]
        ys_train = ys[:, idx]

        print(f"Training for {n_steps} steps on {frac_len * 100:.1f}% of each trajectory")
        print(f"Using {n_points} time points out of {traj_len} total")
        print(f"Starting training loop...")

        for step, (y_batch,) in zip(range(n_steps), dataloader((ys_train,), config.batch_size, key=loader_key)):
            loss, model, opt_state = training_step(ts_train, y_batch, model, opt_state)

            if step % config.print_every == 0 or step == n_steps - 1:
                t_now = time.time()
                total_elapsed = t_now - t_global_start
                print(f"[Step {step:>4}] Loss: {loss:.6f} | Total time elapsed: {total_elapsed:.2f}s")

        print(f"✅ Phase {phase_idx + 1} completed!")

        # Save intermediate model after each phase
        if config.save_intermediate and config.save_final and len(config.steps_strategy) > 1:
            intermediate_path = config.model_save_path.replace('.pkl', f'_phase_{phase_idx + 1}.pkl')
            save_model(model, intermediate_path)

    # Detailed comparison for a specific trajectory
    if config.anim:
        create_comparison_animation(model, ts, ys, config)

    # Plot trajectories
    if config.plot:
        # Generate new unseen test data
        test_key = jr.split(data_key, 1)[0]
        test_config = Config(dataset_size=config.plot_samples)
        ts_test, ys_test = get_data(test_config, key=test_key)

        # Use the training data for the left subplot
        ts_train_plot = ts_train if 'ts_train' in locals() else ts
        ys_train_plot = ys_train if 'ys_train' in locals() else ys

        plot_trajectories(model, ts_train_plot, ys_train_plot, ts_test, ys_test, config)
    else:
        print("\nSkipping plot generation")

    print(f"\n✅ Training completed successfully!")
    print(f"Total time: {time.time() - t_global_start:.2f} seconds")

    # Save final model
    if config.save_final:
        save_model(model, config.model_save_path)

    return ts, ys, model


def debug_tree(pytree, name):
    """Debug function to inspect PyTree structure."""

    def _debug(x, path=()):
        print(f"{name} at {path}: {type(x)}")
        return x

    jax.tree_util.tree_map_with_path(_debug, pytree)


if __name__ == "__main__":
    print("\nNeural ODE Training Script")
    print("=" * 50)

    # Create default config files if they don't exist
    if not os.path.exists("configs"):
        create_default_configs()

    # Example usage with different scenarios:

    # Scenario 1: Train from scratch with default config
    # main()

    # Scenario 2: Use JSON configuration file
    # main(config_path="configs/high_quality.json")

    # Scenario 3: Load existing model for inference
    main(config_path="configs/inference.json")
