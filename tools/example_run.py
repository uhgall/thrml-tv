#!/usr/bin/env python3
"""
Example execution script for the Potts interference sampler using toy data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from thrml.block_sampling import sample_blocks

from graph import conflict_count, potts_energy, random_initial_colors
from model import (
    block_state_to_colors,
    build_potts_coloring_model,
    colors_to_block_state,
)
try:  # pragma: no cover - allow running as script or module
    from .generate_subgraph import load_tv_graph
except ImportError:  # pragma: no cover
    from tools.generate_subgraph import load_tv_graph
from visualization import render_force_layout_html


def run_example(
    *,
    dataset: str | Path = "default",
    stop_energy: float = 0.0,
    output_html_path: str | Path | None = None,
    template_path: str | Path | None = None,
    dataset_name: str | None = None,
    edge_penalty: float = 1.0,
    domain_penalty: float = 15.0,
    steps: int = 2000,
    log_every: int = 20,
    seed: int = 4269,
) -> None:
    """
    Run a Potts-model interference sampler using a dataset under ``input/``.

    Parameters
    ----------
    dataset:
        Subdirectory inside ``input/`` to load (defaults to ``input/default``).
    stop_energy:
        Halt sampling early once the instantaneous energy is less than or equal to this threshold.
    output_html_path:
        Destination path for a self-contained d3-force HTML animation. When omitted, the name is inferred
        from ``dataset_name`` (`force_animation_<dataset>.html`).
    template_path:
        Optional override for the HTML template used to render the visualization.
    dataset_name:
        Optional label for output artifacts; defaults to the dataset directory name.
    edge_penalty:
        Scale factor applied to the pairwise Potts edge penalties.
    domain_penalty:
        Penalty multiplier applied when assignments violate station-channel domains.
    steps:
        Maximum number of block-sampling iterations to run.
    log_every:
        Print progress every ``log_every`` iterations.
    seed:
        Random seed used for JAX PRNG initialisation.
    """

    dataset_path = Path(dataset)
    graph, _ = load_tv_graph(dataset_path)

    dataset_name = dataset_name or (dataset_path.name if dataset_path.name else "graph")

    if output_html_path is None:
        output_html_path = f"force_animation_{dataset_name}.html"

    sampling_program, nodes, potts_data = build_potts_coloring_model(
        graph,
        edge_penalty=edge_penalty,
        domain_penalty=domain_penalty,
    )

    station_ids = potts_data["station_ids"]
    channel_values = potts_data["channel_values"]
    domain_mask_np = potts_data["domain_mask"]
    edges_np = potts_data["edges"]
    edge_weights_np = potts_data["edge_weights"]
    edge_tags = potts_data["edge_tags"]

    N = station_ids.shape[0]
    K = channel_values.shape[0]

    free_blocks = sampling_program.gibbs_spec.free_blocks
    domain_mask = jnp.asarray(domain_mask_np, dtype=bool)

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    colors = random_initial_colors(init_key, N, K, domain_mask)
    block_state = colors_to_block_state(colors, free_blocks)
    sampler_states = [sampler.init() for sampler in sampling_program.samplers]

    edges_jax = jnp.asarray(edges_np, dtype=jnp.int32)
    edge_weights_jax = jnp.asarray(edge_weights_np, dtype=jnp.float32)

    node_indices = jnp.arange(N)

    def total_energy(color_assignments: jnp.ndarray) -> jnp.ndarray:
        potts = potts_energy(color_assignments, edges_jax, edge_weights_jax, edge_penalty)
        domain_violation = jnp.logical_not(domain_mask[node_indices, color_assignments])
        penalty = domain_penalty * jnp.sum(domain_violation)
        return potts + penalty

    initial_energy = float(total_energy(colors))
    initial_conflicts = int(conflict_count(colors, edges_jax, edge_weights_jax))
    initial_domain_violations = int(
        jnp.sum(jnp.logical_not(domain_mask[node_indices, colors]))
    )
    edge_indices = jnp.arange(edges_np.shape[0], dtype=jnp.int32)

    def edge_conflict_mask_for(color_assignments: jnp.ndarray) -> np.ndarray:
        if edges_np.shape[0] == 0:
            return np.zeros((0,), dtype=int)
        penalties = edge_weights_jax[
            edge_indices,
            color_assignments[edges_jax[:, 0]],
            color_assignments[edges_jax[:, 1]],
        ]
        mask = (penalties > 0.0).astype(jnp.int32)
        return np.asarray(mask, dtype=int)

    best_colors = colors
    best_energy = initial_energy
    best_conflicts = initial_conflicts
    best_domain_violations = initial_domain_violations

    print(
        "init energy:",
        best_energy,
        "edge conflicts:",
        best_conflicts,
        "domain violations:",
        best_domain_violations,
    )

    history: list[dict[str, object]] = [
        {
            "step": 0,
            "energy": initial_energy,
            "edgeConflicts": initial_conflicts,
            "domainViolations": initial_domain_violations,
            "colors": np.asarray(colors, dtype=int).tolist(),
            "conflictMask": edge_conflict_mask_for(colors).tolist(),
        }
    ]

    for step in range(1, steps + 1):
        key, subkey = jax.random.split(key)
        block_state, sampler_states = sample_blocks(
            subkey,
            block_state,
            [],
            sampling_program,
            sampler_states,
        )
        colors = block_state_to_colors(block_state).astype(jnp.int32)

        energy = float(total_energy(colors))
        conflicts = int(conflict_count(colors, edges_jax, edge_weights_jax))
        domain_violations = int(
            jnp.sum(jnp.logical_not(domain_mask[node_indices, colors]))
        )

        history.append(
            {
                "step": step,
                "energy": energy,
                "edgeConflicts": conflicts,
                "domainViolations": domain_violations,
                "colors": np.asarray(colors, dtype=int).tolist(),
                "conflictMask": edge_conflict_mask_for(colors).tolist(),
            }
        )

        if energy < best_energy:
            best_energy = energy
            best_conflicts = conflicts
            best_domain_violations = domain_violations
            best_colors = colors

        if step % log_every == 0 or step == steps:
            print(
                f"step {step:04d} energy: {energy:.2f} edge conflicts: {conflicts} "
                f"domain violations: {domain_violations}"
            )

        if energy <= stop_energy:
            print(f"Reached stop_energy={stop_energy} at step {step}; stopping early.")
            break

    print("\nbest summary")
    print(
        "energy:",
        best_energy,
        "edge conflicts:",
        best_conflicts,
        "domain violations:",
        best_domain_violations,
    )

    best_channel_numbers = [channel_values[int(color)] for color in np.array(best_colors)]
    assignment = list(zip(station_ids.tolist(), best_channel_numbers))
    for station_id, channel in assignment:
        print(f"station {station_id}: channel {channel}")

    save_force_layout_animation(
        output_html_path=output_html_path,
        station_ids=station_ids,
        edges=edges_np,
        channel_values=channel_values,
        frames=history,
        edge_tags=edge_tags,
        dataset_name=dataset_name,
        template_path=template_path,
    )

    print(f"\nSaved D3 animation to {Path(output_html_path).resolve()}")


def save_force_layout_animation(
    output_html_path: str | Path,
    station_ids: np.ndarray,
    edges: np.ndarray,
    channel_values: np.ndarray,
    frames: list[dict[str, object]],
    *,
    edge_tags: list[list[str]],
    dataset_name: str,
    template_path: str | Path | None = None,
) -> None:
    """
    Build the payload required for the force-layout visualization and render it via the HTML template.
    """

    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    node_payload = [
        {"id": int(idx), "stationId": int(station_ids[idx])}
        for idx in range(station_ids.shape[0])
    ]

    edges_list = np.asarray(edges, dtype=np.int32).tolist()
    edge_payload = []
    for edge, tags in zip(edges_list, edge_tags):
        edge_payload.append(
            {
                "source": int(edge[0]),
                "target": int(edge[1]),
                "typeTags": list(tags),
            }
        )

    data_payload = {
        "datasetName": dataset_name,
        "nodes": node_payload,
        "links": edge_payload,
        "channelLabels": [int(ch) for ch in np.asarray(channel_values, dtype=np.int32)],
        "frames": frames,
    }

    render_force_layout_html(data_payload, output_path, template_path=template_path)


def _build_cli_parser() -> argparse.ArgumentParser:
    """
    Construct the CLI parser used for invoking the example sampler.
    """
    parser = argparse.ArgumentParser(
        description="Run the Potts interference sampler on a dataset under ./input/."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("default"),
        help="Subdirectory under ./input/ containing Domain.csv, Interference_Paired.csv, and parameters.csv (default: default).",
    )
    parser.add_argument(
        "--stop-energy",
        type=float,
        default=0.0,
        help="Stop once the instantaneous energy falls below this threshold (default: 0.0).",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Destination path for the generated force-layout animation HTML.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Optional override for the force-layout HTML template.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Override label used in logs and generated artifacts.",
    )
    parser.add_argument(
        "--edge-penalty",
        type=float,
        default=1.0,
        help="Scale factor applied to edge penalties (default: 1.0).",
    )
    parser.add_argument(
        "--domain-penalty",
        type=float,
        default=15.0,
        help="Penalty multiplier applied to domain violations (default: 15.0).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Maximum number of Gibbs sampling steps to run (default: 2000).",
    )
    parser.add_argument(
        "--log-interval",
        dest="log_every",
        type=int,
        default=20,
        help="Print progress every N steps (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4269,
        help="Random seed used for sampler initialisation (default: 4269).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry-point compatible with other tools in this package.
    """
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    run_example(
        dataset=args.input,
        stop_energy=args.stop_energy,
        output_html_path=args.output_html,
        template_path=args.template,
        dataset_name=args.dataset_name,
        edge_penalty=args.edge_penalty,
        domain_penalty=args.domain_penalty,
        steps=args.steps,
        log_every=args.log_every,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

