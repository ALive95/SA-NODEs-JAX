import equinox as eqx
import diffrax
import jax.numpy as jnp
import jax
import jax.nn as jnn
from typing import Sequence, Union

# --- Activation Functions ---
# Use JAX's built-in ReLU for simplicity and efficiency
# def custom_activation(x):
#     return jnp.sin(x) ** 2 + x # Original custom activation


# --- Initialization Functions ---
def soft_init(key: jax.random.PRNGKey, shape: Sequence[int], scale: float = 0.01) -> jnp.ndarray:
    """
    Soft initialization: small random values from a normal distribution.
    This helps prevent exploding gradients, especially with ReLU.
    """
    return jax.random.normal(key, shape) * scale


# --- Custom Layers ---
class Linear(eqx.Module):
    """A standard linear layer (fully connected layer)."""
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_features: int, out_features: int, *, key: jax.random.PRNGKey, init_scale: float = 0.01):
        """
        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            key: JAX PRNGKey for initialization.
            init_scale: Scaling factor for weight initialization (e.g., for soft_init).
        """
        key_w, key_b = jax.random.split(key)
        self.weight = soft_init(key_w, (out_features, in_features), scale=init_scale)
        # Initialize bias to zeros, as is common practice
        self.bias = jnp.zeros((out_features,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the linear transformation.
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        return self.weight @ x + self.bias


class MLP(eqx.Module):
    """A simple Multi-Layer Perceptron (MLP)."""
    layers: list
    activation: callable

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        *,
        key: jax.random.PRNGKey,
        init_scale: float = 0.01,
        activation: callable = jnn.relu  # Default to ReLU
    ):
        """
        Args:
            in_size: Dimension of the input.
            out_size: Dimension of the output.
            width_size: Number of neurons in hidden layers.
            depth: Number of hidden layers. (Total layers = depth + 1)
            key: JAX PRNGKey for initialization.
            init_scale: Scaling factor for weight initialization.
            activation: Activation function to use between layers.
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative.")
        if depth == 0: # Handle depth 0 for direct input-output mapping
            self.layers = [Linear(in_size, out_size, key=key, init_scale=init_scale)]
            self.activation = lambda x: x # No activation for single layer
        else:
            keys = jax.random.split(key, depth + 1)
            self.activation = activation
            self.layers = []

            # Input layer: in_size -> width_size
            self.layers.append(Linear(in_size, width_size, key=keys[0], init_scale=init_scale))

            # Hidden layers: width_size -> width_size
            for i in range(depth - 1):
                self.layers.append(Linear(width_size, width_size, key=keys[i + 1], init_scale=init_scale))

            # Output layer: width_size -> out_size
            self.layers.append(Linear(width_size, out_size, key=keys[-1], init_scale=init_scale))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a forward pass through the MLP.
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        # Apply activation to all but the last layer
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        # The last layer typically does not have an activation in MLPs for regression tasks
        return self.layers[-1](x)


class ODEFunc(eqx.Module):
    """
    Represents the differential function f(t, y, args) in dy/dt = f(t, y, args).
    This function is approximated by an MLP.
    """
    mlp: MLP

    def __init__(self, data_size: int, width_size: int, depth: int, *, key: jax.random.PRNGKey, init_scale: float = 0.01):
        """
        Args:
            data_size: Dimension of the state vector y.
            width_size: Width of the hidden layers in the MLP.
            depth: Number of hidden layers in the MLP.
            key: JAX PRNGKey for MLP initialization.
            init_scale: Scaling factor for weight initialization.
        """
        super().__init__()
        # The MLP takes `y` and `t` as input, so its input size is data_size + 1.
        # It outputs the derivative `dy/dt`, which has the same size as `y`.
        self.mlp = MLP(
            in_size=data_size + 1,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            key=key,
            init_scale=init_scale,
            activation=jnn.relu, # Explicitly use ReLU
        )

    def __call__(self, t: float, y: jnp.ndarray, args: None = None) -> jnp.ndarray:
        """
        Computes f(t, y, args) for the ODE.
        Args:
            t: Current time.
            y: Current state vector.
            args: Additional arguments (not used here, but required by Diffrax).
        Returns:
            The derivative dy/dt.
        """
        # Concatenate time `t` as an additional input feature to the state `y`.
        # t is a scalar, so convert it to an array for concatenation.
        y_and_t = jnp.concatenate([y, jnp.array([t])], axis=0)
        return self.mlp(y_and_t)


class NeuralODE(eqx.Module):
    """
    A Neural Ordinary Differential Equation model.
    It wraps the `ODEFunc` and uses Diffrax to solve the ODE.
    """
    func: ODEFunc

    def __init__(self, data_size: int, width_size: int, depth: int, *, key: jax.random.PRNGKey):
        """
        Args:
            data_size: Dimension of the state vector y.
            width_size: Width of the hidden layers in the ODE function MLP.
            depth: Number of hidden layers in the ODE function MLP.
            key: JAX PRNGKey for initialization.
        """
        self.func = ODEFunc(data_size, width_size, depth, key=key)

    def __call__(self, ts: jnp.ndarray, y0: jnp.ndarray) -> jnp.ndarray:
        """
        Solves the ODE from an initial condition `y0` over a sequence of time points `ts`.
        Args:
            ts: A 1D JAX array of time points at which to save the solution.
                Must be sorted in increasing order.
            y0: The initial state vector at `ts[0]`.
        Returns:
            A JAX array of shape `(len(ts), data_size)` containing the solution
            `y` at each time point in `ts`.
        """
        # Define the ODE term using the learned function
        term = diffrax.ODETerm(self.func)

        # Choose a solver. Dopri5 is a good general-purpose choice.
        solver = diffrax.Dopri5()

        # Specify when to save the solution
        saveat = diffrax.SaveAt(ts=ts)

        # Configure the step size controller for adaptive stepping
        # PIDController is robust for most problems.
        # rtol (relative tolerance) and atol (absolute tolerance) control accuracy.
        controller = diffrax.PIDController(rtol=1e-3, atol=1e-4)

        # Ensure t0, t1, and dt0 are correctly set based on ts
        t0 = ts[0]
        t1 = ts[-1]
        # dt0 can be inferred from the first two time points, or set small
        # It's an initial guess for the step size.
        dt0 = ts[1] - ts[0] if len(ts) > 1 else 0.1 # Fallback for single time point

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            saveat=saveat,
            stepsize_controller=controller,
            # args=None, # args can be passed if self.func uses them, explicit here for clarity
        )

        return solution.ys