# SA-NODEs with JAX

A JAX-based implementation for training Semi-Autonomous Neural Ordinary Differential Equations (SA-NODEs).
This is based on the paper 

"Universal Approximation of Dynamical Systems by Semi-Autonomous Neural ODEs and Applications",

by Ziqian Li and Kang Liu and Lorenzo Liverani (myself) and Enrique Zuazua.
    

## Overview

This program implements a SA-NODE system that learns to approximate the dynamics of a 2D or 3D ordinary 
differential equation. The system uses JAX for automatic differentiation and just-in-time compilation, 
Equinox for neural network components, and Diffrax for ODE solving.

## Features

- **Configurable Training Pipeline**: JSON-based configuration system for reproducible experiments
- **Multi-Phase Training**: Progressive training with different trajectory lengths and step counts
- **Model Persistence**: Save and load trained models with Equinox serialization
- **Visualization**: Trajectory plotting and detailed animation generation
- **Regularization**: L2 regularization with configurable strength
- **Batch Processing**: Efficient batch training with random sampling

## Installation

### Requirements

```bash
pip install jax jax-lib
pip install diffrax
pip install equinox
pip install optax
pip install matplotlib  # For plotting utilities
```

### Dependencies

- `jax`: Automatic differentiation and XLA compilation
- `jax-lib`: JAX backend
- `diffrax`: Differential equation solving
- `equinox`: Neural network library for JAX
- `optax`: Optimization library
- `matplotlib`: Visualization (required by plot_utils)

## Quick Start

### 1. Basic Usage

```python
from neural_ode_trainer import main

# Train with default configuration
main()
```

### 2. Using Configuration Files

```python
# Create default configuration files
from neural_ode_trainer import create_default_configs
create_default_configs()

# Train with specific configuration
main(config_path="configs/high_quality.json")
```

### 3. Custom Configuration

```python
from neural_ode_trainer import Config, main

# Create custom configuration
config = Config(
    dataset_size=200,
    batch_size=50,
    learning_rate=1e-3,
    steps_strategy=(1000, 2000),
    length_strategy=(0.3, 0.8),
    width_size=100,
    depth=2
)

# Train with custom config
main(config=config)
```

## Configuration

The program uses a comprehensive configuration system through the `Config` dataclass:

### Dataset Parameters
- `dataset_size`: Number of trajectories to generate (default: 100)
- `time_grid_start/end`: Time range for ODE integration (default: 0.0 to 5.0)
- `time_grid_points`: Number of time points per trajectory (default: 200)

### Training Parameters
- `batch_size`: Training batch size (default: 100)
- `learning_rate`: Initial learning rate (default: 5e-4)
- `steps_strategy`: Tuple of training steps per phase (default: (3000, 6000))
- `length_strategy`: Fraction of trajectory length per phase (default: (0.5, 1.0))

### Model Parameters
- `width_size`: Hidden layer width (default: 200)
- `depth`: Number of hidden layers (default: 1)
- `lambda_reg`: L2 regularization strength (default: 1e-6)

### Visualization Parameters
- `plot`: Enable trajectory plotting (default: True)
- `anim`: Enable animation generation (default: True)
- `plot_samples`: Number of test trajectories to plot (default: 100)

## Default Configuration Files

The program creates several preset configurations:

### `configs/default.json`
Standard training configuration with balanced parameters.

### `configs/fast.json`
Quick training for testing and development:
- Smaller dataset (50 samples)
- Fewer training steps (100, 500)
- Smaller network (50 width, 1 depth)

### `configs/high_quality.json`
High-quality training for production:
- Larger dataset (500 samples)
- More training steps (2000, 5000)
- Larger network (300 width, 3 depth)

### `configs/inference.json`
Inference-only configuration:
- Loads existing model
- No training steps
- Plotting enabled

## Training Process

### Multi-Phase Training

The training occurs in multiple phases, each with different trajectory lengths:

1. **Phase 1**: Train on partial trajectories (e.g., 50% of time points)
2. **Phase 2**: Train on longer trajectories (e.g., 100% of time points)

This progressive approach helps the model learn from simple to complex dynamics.
One can choose to increase or decrease the number of phases, as well as
the percentage of time points in each phase.

### Loss Function

The loss combines:
- **MSE Loss**: Mean squared error between predicted and true trajectories
- **L2 Regularization**: Prevents overfitting by penalizing large weights

```
Total Loss = MSE + λ × L2_regularization
```

### Optimization

- **Optimizer**: Adam with exponential learning rate decay
- **Schedule**: Staircase decay every `transition_steps`
- **Decay Rate**: Configurable exponential decay (default: 0.9)

## Model Architecture

The SA-NODE uses a feed-forward network to approximate the ODE function:

```
f(t, y) = W relu(A y + B t + C) → dy/dt
```

Where:
- Input: Current state `y` (2D for oscillator dynamics)
- Output: Time derivative `dy/dt`
- Architecture: Configurable width and depth

## Ground Truth Dynamics

The system learns to approximate a certain ODE.
Default is the Van Der Pol oscillator:

```
dy/dt = [y₁, (1-y₀²)y₁ - y₀]
```

Where:
- `y₀`: Position coordinate
- `y₁`: Velocity coordinate

## File Structure

```
project/
├── main.py    # Main training script
├── jax_models/
│   └── neural_odes.py       # SA-NODE model definition
├── tools/
│   └── plot_utils.py        # Visualization utilities
├── configs/                 # Configuration files
│   ├── default.json
│   ├── fast.json
│   ├── high_quality.json
│   └── inference.json
├── models/                  # Saved models
│   └── neural_ode_model.pkl
└── figures/                 # Generated plots and animations
    ├── jax_Compare.png
    └── detailed_animation.gif
```

## Model Persistence

### Saving Models

Models are automatically saved using Equinox serialization:

```python
# Automatic saving after training
config = Config(save_final=True, model_save_path="models/my_model.pkl")
main(config=config)
```

### Loading Models

```python
# Load existing model for inference
config = Config(load_existing=True, model_save_path="models/my_model.pkl")
main(config=config)
```

## Visualization

### Trajectory Plots

The program generates comparison plots showing:
- **Left Panel**: Training trajectories and model predictions
- **Right Panel**: Test trajectories and model predictions
- Color-coded trajectories with transparency

### Animations

Detailed animations show:
- True vs predicted trajectories over time
- Phase space visualization
- Prediction accuracy evolution

## Usage Examples

### Example 1: Quick Testing

```python
# Fast training for development
main(config_path="configs/fast.json")
```

### Example 2: Production Training

```python
# High-quality training
main(config_path="configs/high_quality.json")
```

### Example 3: Model Evaluation

```python
# Load and evaluate existing model
main(config_path="configs/inference.json")
```

### Example 4: Custom Training

```python
config = Config(
    dataset_size=300,
    steps_strategy=(2000, 4000, 6000),
    length_strategy=(0.3, 0.6, 1.0),
    width_size=150,
    depth=3,
    learning_rate=1e-3,
    lambda_reg=1e-5
)
main(config=config)
```

## Output

The program generates:

1. **Console Output**: Training progress, loss values, timing information
2. **Saved Models**: Pickled model files in `models/` directory
3. **Plots**: Trajectory comparisons in `figures/` directory
4. **Animations**: Detailed dynamics visualization as GIF files

## Performance Tips

1. **GPU Acceleration**: Ensure JAX is configured with GPU support for faster training
2. **Batch Size**: Adjust based on available memory
3. **JIT Compilation**: First few iterations may be slow due to compilation
4. **Regularization**: Tune `lambda_reg` to balance fitting and generalization

### Debug Mode

Use the `debug_tree` function to inspect model structure:

```python
debug_tree(model, "Model Structure")
```

## License

This code is public and freely available.

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@misc{SANODEs,
      title={Universal Approximation of Dynamical Systems by Semi-Autonomous Neural ODEs and Applications}, 
      author={Ziqian Li and Kang Liu and Lorenzo Liverani and Enrique Zuazua},
      year={2024},
      eprint={2407.17092},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2407.17092}, 
}
```
