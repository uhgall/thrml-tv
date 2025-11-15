"""
Example execution script for the Potts interference sampler using toy data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from thrml.block_sampling import sample_blocks

from graph import (
    conflict_count,
    load_domain_csv,
    load_interference_csv,
    potts_energy,
    random_initial_colors,
)
from model import (
    block_state_to_colors,
    build_potts_coloring_model,
    colors_to_block_state,
)
from visualization import render_force_layout_html


def run_example(
    stop_energy: float = 0.0,
    output_html_path: str | None = None,
    *,
    data_dir: str | Path = "input/test4",
    symmetrize_constraints: bool = False,
    template_path: str | Path | None = None,
) -> None:
    """
    Run a Potts-model interference sampler on one of the bundled toy data sets.

    Parameters
    ----------
    stop_energy:
        Halt sampling early once the instantaneous energy is less than or equal to this threshold.
    output_html_path:
        Destination path for a self-contained d3-force HTML animation. When omitted, the name is inferred
        from the data directory (`force_animation_<dataset>.html`).
    data_dir:
        Directory containing `Domain.csv` and `Interference_Paired.csv`.
    symmetrize_constraints:
        If True, directional constraints (ADJ+1 / ADJ-1) are automatically mirrored.
    template_path:
        Optional override for the HTML template used to render the visualization.
    """

    lam = 1.0
    domain_penalty = 15.0
    steps = 2000
    log_every = 20
    seed = 4269

    base_dir = Path(data_dir)
    dataset_name = base_dir.name

    if output_html_path is None:
        output_html_path = f"force_animation_{dataset_name}.html"

    domain_csv_path = base_dir / "Domain.csv"
    interference_csv_path = base_dir / "Interference_Paired.csv"

    station_ids, channel_values, domain_mask_np = load_domain_csv(str(domain_csv_path))
    edges_np, edge_weights_np, edge_tags = load_interference_csv(
        str(interference_csv_path),
        station_ids,
        channel_values,
        symmetrize_constraints=symmetrize_constraints,
        return_edge_metadata=True,
    )

    N = station_ids.shape[0]
    K = channel_values.shape[0]

    sampling_program, nodes = build_potts_coloring_model(
        station_ids=station_ids,
        channel_values=channel_values,
        edges_np=edges_np,
        edge_weights_np=edge_weights_np,
        lam=lam,
        domain_mask_np=domain_mask_np,
        domain_penalty=domain_penalty,
    )

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
        potts = potts_energy(color_assignments, edges_jax, edge_weights_jax, lam)
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


if __name__ == "__main__":
    run_example(data_dir="input/test10", symmetrize_constraints=True)

