from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "initial_particles_pos_env_0.txt"
    default_output = script_dir / "initial_particles_pos_env_0.png"

    parser = argparse.ArgumentParser(
        description="Plot stored initial particle positions as a 3D point cloud."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to the txt file that stores the particle position list.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to save the rendered figure.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Initial particle positions",
        help="Figure title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively instead of only saving it.",
    )
    return parser.parse_args()


def load_positions(input_path: Path) -> np.ndarray:
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    positions = np.asarray(data, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Expected an array with shape (N, 3), but got {positions.shape}."
        )
    return positions


def set_equal_axes(ax, positions: np.ndarray) -> None:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    centers = (mins + maxs) / 2.0
    half_range = float(np.max(maxs - mins) / 2.0)
    if half_range <= 0.0:
        half_range = 1.0

    ax.set_xlim(centers[0] - half_range, centers[0] + half_range)
    ax.set_ylim(centers[1] - half_range, centers[1] + half_range)
    ax.set_zlim(centers[2] - half_range, centers[2] + half_range)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def plot_positions(
    positions: np.ndarray,
    output_path: Path,
    title: str,
    show: bool,
) -> None:
    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=18,
        c=positions[:, 2],
        cmap="Reds",
        alpha=0.85,
        depthshade=True,
    )
    fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08, label="z")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    set_equal_axes(ax, positions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    positions = load_positions(args.input)
    plot_positions(
        positions=positions,
        output_path=args.output,
        title=args.title,
        show=args.show,
    )
    print(f"Saved particle plot to: {args.output}")


if __name__ == "__main__":
    main()
