import json
from main import Config
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import numpy as np
import matplotlib.gridspec as gridspec


def plot_trajectories(model, ts_train, ys_train, ts_test, ys_test, config: Config):
    """
    Plot training and testing trajectories with model predictions.

    Args:
        model: Trained Neural ODE model
        ts_train: Training time points
        ys_train: Training trajectory data
        ts_test: Test time points
        ys_test: Test trajectory data
        config: Configuration object containing plot parameters
    """
    print(f"\nGenerating plots with {config.plot_samples} test samples...")

    dim = ys_test.shape[2]  # Automatically detect the dimension
    print(f"Creating visualization for {dim}D system...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.plot_figsize)

    if dim == 3:
        # Remove the existing axes and create 3D ones
        ax1.remove()
        ax2.remove()
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        print("Plotting 3D trajectories...")

        # Left subplot: Training data vs predictions
        n_train_plot = min(config.plot_max_trajectories, ys_train.shape[0])
        for i in range(n_train_plot):
            ax1.plot(ys_train[i, :, 0], ys_train[i, :, 1], ys_train[i, :, 2],
                     c="blue", alpha=0.6, label="Training Data" if i == 0 else None)
            pred_train = model(ts_train, ys_train[i, 0])
            ax1.plot(pred_train[:, 0], pred_train[:, 1], pred_train[:, 2],
                     c="red", alpha=0.6, label="Prediction" if i == 0 else None)

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("Training Data vs Predictions")
        ax1.legend()

        # Right subplot: New unseen data vs predictions
        n_test_plot = min(config.plot_max_trajectories, config.plot_samples)
        for i in range(n_test_plot):
            ax2.plot(ys_test[i, :, 0], ys_test[i, :, 1], ys_test[i, :, 2],
                     c="blue", alpha=1, label="Unseen Data" if i == 0 else None)
            pred_test = model(ts_test, ys_test[i, 0])
            ax2.plot(pred_test[:, 0], pred_test[:, 1], pred_test[:, 2],
                     c="green", alpha=1, label="Prediction" if i == 0 else None)

        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.set_title("Unseen Data vs Predictions")
        ax2.legend()

    else:
        print("Plotting 2D phase space trajectories...")

        # Left subplot: Training data vs predictions
        n_train_plot = min(config.plot_max_trajectories, ys_train.shape[0])
        for i in range(n_train_plot):
            ax1.plot(ys_train[i, :, 0], ys_train[i, :, 1],
                     c="blue", alpha=1, label="Training Data" if i == 0 else None)
            pred_train = model(ts_train, ys_train[i, 0])
            ax1.plot(pred_train[:, 0], pred_train[:, 1],
                     c="red", alpha=0.6, label="Prediction" if i == 0 else None)

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("Training")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right subplot: New unseen data vs predictions
        n_test_plot = min(config.plot_max_trajectories, config.plot_samples)
        for i in range(n_test_plot):
            ax2.plot(ys_test[i, :, 0], ys_test[i, :, 1],
                     c="blue", alpha=1, label="Testing Data" if i == 0 else None)
            pred_test = model(ts_test, ys_test[i, 0])
            ax2.plot(pred_test[:, 0], pred_test[:, 1],
                     c="green", alpha=1, label="Prediction" if i == 0 else None)

        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Testing")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config.plot_save_path), exist_ok=True)

    print(f"Saving plot to {config.plot_save_path}...")
    plt.savefig(config.plot_save_path, dpi=config.plot_dpi, bbox_inches='tight')
    plt.show()
    print("✅ Plotting completed!")


def create_comparison_animation(model, ts_test, ys_test, config: Config):
    """
    Create a focused animation showing one trajectory with detailed comparison.

    Args:
        model: Trained Neural ODE model
        ts_test: Test time points
        ys_test: Test trajectory data
        config: Configuration object
    """
    print(f"\nCreating detailed comparison animation for trajectory {config.trajectory_idx}...")

    dim = ys_test.shape[2]
    n_frames = len(ts_test)

    # Taking parameters from config
    fps = config.fps
    interval = config.interval
    trajectory_idx = config.trajectory_idx
    animation_save_path = config.animation_save_path

    # Generate prediction for the selected trajectory
    prediction = model(ts_test, ys_test[trajectory_idx, 0])

    # Calculate errors
    errors = np.abs(ys_test[trajectory_idx] - prediction)

    # Adjust figure size based on dimension for better viewing
    if dim == 3:
        fig = plt.figure(figsize=(22, 14)) # Slightly larger figure for better spacing
        # height_ratios: [1, 1, 1] for the 3 time plots, and [1.0] for the taller error plot
        gs = gridspec.GridSpec(dim + 1, 2, height_ratios=[1]*dim + [1.0], width_ratios=[1, 1])
    else: # dim == 2
        fig = plt.figure(figsize=(20, 10)) # Slightly larger figure for better spacing
        # height_ratios: [1, 1] for the 2 time plots, and [1.0] for the taller error plot
        gs = gridspec.GridSpec(dim + 1, 2, height_ratios=[1]*dim + [1.0], width_ratios=[1, 1])

    # Set the overall title for the animation
    fig.suptitle("Summary Animation", fontsize=16)

    # --- Phase space subplot (left square) ---
    # This subplot should span from row 0 up to (but not including) the last row (dim)
    # in the first column (index 0).
    ax1 = fig.add_subplot(gs[0:dim, 0], projection='3d' if dim == 3 else None)
    ax1.set_title("Phase Space Trajectory")

    # Set common limits for phase space
    x_min, x_max = min(ys_test[trajectory_idx, :, 0].min(), prediction[:, 0].min()), \
                   max(ys_test[trajectory_idx, :, 0].max(), prediction[:, 0].max())
    y_min, y_max = min(ys_test[trajectory_idx, :, 1].min(), prediction[:, 1].min()), \
                   max(ys_test[trajectory_idx, :, 1].max(), prediction[:, 1].max())

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    if dim == 3:
        z_min, z_max = min(ys_test[trajectory_idx, :, 2].min(), prediction[:, 2].min()), \
                       max(ys_test[trajectory_idx, :, 2].max(), prediction[:, 2].max())
        ax1.set_zlim(z_min, z_max)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        # True trajectory is FIXED and plotted from the very beginning
        ax1.plot(ys_test[trajectory_idx, :, 0], ys_test[trajectory_idx, :, 1],
                 ys_test[trajectory_idx, :, 2], 'b-', linewidth=3, label="True", alpha=0.8)
        line_pred_phase, = ax1.plot([], [], [], 'r--', linewidth=2, label="Prediction", alpha=0.8)
        point_true, = ax1.plot([], [], [], 'bo', markersize=8, label="Current True")
        point_pred, = ax1.plot([], [], [], 'ro', markersize=8, label="Current Pred")
    else: # dim == 2
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.grid(True, alpha=0.3)

        # True trajectory is FIXED and plotted from the very beginning
        ax1.plot(ys_test[trajectory_idx, :, 0], ys_test[trajectory_idx, :, 1],
                 'b-', linewidth=3, label="True", alpha=0.8)
        line_pred_phase, = ax1.plot([], [], 'r--', linewidth=2, label="Prediction", alpha=0.8)
        point_true, = ax1.plot([], [], 'bo', markersize=8, label="Current True")
        point_pred, = ax1.plot([], [], 'ro', markersize=8, label="Current Pred")

    ax1.legend()

    # --- Time evolution subplots (right column) ---
    axes_time_evolution = []
    lines_pred_time = [] # Only need to update prediction lines
    points_pred_time = [] # New: for point_pred on time evolution plots
    colors = ['orange', 'green', 'purple'] # Define colors for each dimension
    labels = ['x', 'y', 'z'] # Define labels for each dimension

    for d in range(dim):
        # Each time evolution subplot occupies a row (d) in the second column (index 1)
        ax_time = fig.add_subplot(gs[d, 1])
        ax_time.set_title(f"Time Evolution of {labels[d]}")
        ax_time.set_xlabel("Time")
        ax_time.grid(True, alpha=0.3)
        ax_time.set_xlim(ts_test[0], ts_test[-1])

        # Set y-limits based on all values for the specific dimension
        min_val = min(ys_test[trajectory_idx, :, d].min(), prediction[:, d].min())
        max_val = max(ys_test[trajectory_idx, :, d].max(), prediction[:, d].max())
        ax_time.set_ylim(min_val, max_val)

        # True solution is fixed and plotted from the very beginning
        ax_time.plot(ts_test, ys_test[trajectory_idx, :, d], color=colors[d], linewidth=2,
                     label=f"True {labels[d]}", alpha=0.8)
        line_pred, = ax_time.plot([], [], color="red", linewidth=1, linestyle='--',
                                  label=f"Pred {labels[d]}", alpha=0.8)
        point_pred_time, = ax_time.plot([], [], 'ro', markersize=6, label="Current Pred") # New point_pred
        lines_pred_time.append(line_pred)
        points_pred_time.append(point_pred_time) # Append new point_pred artist
        axes_time_evolution.append(ax_time)
        # Place legend outside the plot to prevent misalignment
        ax_time.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        # Remove y-axis label to help with alignment
        ax_time.set_ylabel("")

    # --- Error evolution subplot (bottom) ---
    # This subplot spans both columns in the last row (index dim)
    ax_error = fig.add_subplot(gs[dim, :])
    ax_error.set_title("Prediction Error")
    ax_error.set_xlabel("Time")
    ax_error.set_ylabel("Absolute Error")
    ax_error.grid(True, alpha=0.3)
    ax_error.set_xlim(ts_test[0], ts_test[-1])

    # Set y-axis to log scale
    ax_error.set_yscale('log')
    # Ensure min value is not zero for log scale
    min_error_val = np.min(errors[errors > 0]) # Get the minimum non-zero error
    ax_error.set_ylim(min_error_val * 0.9, errors.max() * 1.1)

    lines_error = []
    for d in range(dim):
        line_error, = ax_error.plot([], [], color=colors[d], linewidth=2,
                                    label=f"Error {labels[d]}", alpha=0.8)
        lines_error.append(line_error)
    ax_error.legend()

    # Use plt.tight_layout with a larger pad for more separation
    # Increased pad, w_pad, and h_pad for more space
    plt.tight_layout(pad=3.0, w_pad=4.0, h_pad=3.0)

    def animate(frame):
        # Update phase space (only prediction and current points, true is fixed)
        if dim == 3:
            line_pred_phase.set_data(prediction[:frame + 1, 0], prediction[:frame + 1, 1])
            line_pred_phase.set_3d_properties(prediction[:frame + 1, 2])

            point_true.set_data([ys_test[trajectory_idx, frame, 0]],
                                [ys_test[trajectory_idx, frame, 1]])
            point_true.set_3d_properties([ys_test[trajectory_idx, frame, 2]])

            point_pred.set_data([prediction[frame, 0]], [prediction[frame, 1]])
            point_pred.set_3d_properties([prediction[frame, 2]])
        else: # dim == 2
            line_pred_phase.set_data(prediction[:frame + 1, 0], prediction[:frame + 1, 1])

            point_true.set_data([ys_test[trajectory_idx, frame, 0]],
                                [ys_test[trajectory_idx, frame, 1]])
            point_pred.set_data([prediction[frame, 0]], [prediction[frame, 1]])

        # Update individual time evolution plots (only prediction lines and points, true is fixed)
        for d in range(dim):
            lines_pred_time[d].set_data(ts_test[:frame + 1], prediction[:frame + 1, d])
            points_pred_time[d].set_data([ts_test[frame]], [prediction[frame, d]]) # Update point_pred

        # Update error evolution
        for d in range(dim):
            lines_error[d].set_data(ts_test[:frame + 1], errors[:frame + 1, d])

        # Return all updated artists for blitting
        # Note: Fixed lines (e.g., true trajectories) are not returned for blitting
        # as they don't change.
        artists = ([line_pred_phase, point_true, point_pred] +
                   lines_pred_time + points_pred_time + lines_error) # Include points_pred_time in artists
        return artists

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=interval,
                                   blit=True, repeat=True)

    # Save animation if path is provided
    if animation_save_path:
        print(f"Saving animation to {animation_save_path}...")
        os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
        anim.save(animation_save_path, writer='pillow', fps=fps, dpi=100)
        print(f"✅ Animation saved to {animation_save_path}")

    print("✅ Animation completed!")

    return anim
