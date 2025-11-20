#!/usr/bin/env python3
"""
Run a THRML Potts sampler on an FCC TVGraph.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from lib.tv_graph import TVGraph
from graph_stats import compute_graph_stats

from thrml import Block, BlockGibbsSpec, FactorSamplingProgram, SamplingSchedule, sample_states
from thrml.block_management import block_state_to_global
from thrml.block_sampling import sample_blocks
from thrml.models import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.models.ebm import DEFAULT_NODE_SHAPE_DTYPES
from thrml.pgm import CategoricalNode

WARNING_PREFIX = "!!! WARNING"

if TYPE_CHECKING:  # pragma: no cover
    from lib.tv_web_viz import WebVizController


@dataclass(slots=True)
class PottsComponents:
    """
    Bundle the THRML artefacts needed for sampling and diagnostics.
    """

    program: FactorSamplingProgram
    factors: list[CategoricalEBMFactor]
    free_blocks: list[Block]
    init_state: list[jnp.ndarray]
    lambda_conflict_initial: float
    lambda_conflict_current: float
    conflict_factor_index: int | None
    conflict_base_weights: jnp.ndarray | None


def _emit_log(message: str, log_fn: Callable[[str], None] | None) -> None:
    """
    Dispatch a status message to the provided logger or stdout.
    """

    if log_fn is not None:
        log_fn(message)
    else:
        print(message)


def _emit_warning(message: str, log_fn: Callable[[str], None] | None) -> None:
    """
    Emit a warning-prefixed log message for higher visibility.
    """

    _emit_log(f"{WARNING_PREFIX}: {message}", log_fn)


def _summarise_graph_stats(graph: TVGraph) -> list[str]:
    """
    Produce concise graph statistics for console and web logging.
    """

    stats = compute_graph_stats(graph)
    domain_violations, interference_violations = graph.assignment_violations()

    summary = [
        f"Stations: {stats['station_count']:,}",
        f"Channels: {stats['channel_count']:,}",
        (
            "Domain size avg/min/max: "
            f"{stats['domain_avg']:.2f} / {stats['domain_min']} / {stats['domain_max']}"
        ),
        (
            "Directional constraints: "
            f"{stats['constraint_total']:,} (avg {stats['constraint_avg']:.2f} per station)"
        ),
        (
            "Constraint degree min/max: "
            f"{stats['constraint_min']} / {stats['constraint_max']}"
        ),
        (
            "Assignment violations — domain: "
            f"{len(domain_violations):,}, interference: {len(interference_violations):,}"
        ),
    ]

    type_counts: Counter[str] = stats["constraint_counts_by_type"]  # type: ignore[assignment]
    if type_counts:
        top_types = ", ".join(
            f"{constraint_type}:{count:,}"
            for constraint_type, count in type_counts.most_common(3)
        )
        summary.append(f"Constraint types: {top_types}")

    top_by_constraints: Sequence[tuple[int, int]] = stats["top_by_constraints"]  # type: ignore[assignment]
    if top_by_constraints:
        top_station_id, top_constraint_count = top_by_constraints[0]
        summary.append(
            "Most constrained station: "
            f"{top_station_id} ({top_constraint_count:,} directional edges)"
        )

    return summary


def _make_domain_factor(
    graph: TVGraph,
    nodes: Sequence[CategoricalNode],
    penalty: float,
    *,
    log_fn: Callable[[str], None] | None = None,
) -> CategoricalEBMFactor:
    """
    Encode station-specific channel domains via a unary ``CategoricalEBMFactor``.

    ``DiscreteEBMFactor`` subtracts the weight tensor from the energy, so we assign ``-penalty`` to illegal channels,
    causing a positive energy bump whenever the sampler leaves the station's domain.
    """

    station_count = len(nodes)
    channel_count = graph.channel_count
    _emit_log(
        f"    Building domain penalty matrix ({station_count} stations × {channel_count} channels).",
        log_fn,
    )
    weights = np.full((station_count, channel_count), -float(penalty), dtype=np.float32)
    for station in graph.stations_by_id.values():
        if station.station_index is None:
            raise ValueError(f"Station {station.station_id} is missing a contiguous index.")
        weights[station.station_index, station.domain_indices] = 0.0
    _emit_log("    Domain penalty matrix populated.", log_fn)
    return CategoricalEBMFactor([Block(nodes)], jnp.asarray(weights))


def _make_interference_factor(
    graph: TVGraph,
    nodes: Sequence[CategoricalNode],
    penalty: float,
    *,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[CategoricalEBMFactor, jnp.ndarray] | None:
    """
    Create a pairwise Potts factor that penalises channel combinations forbidden by FCC constraints.
    Collapse symmetric FCC rows and yield unique incompatibilities.
    """

    _emit_log("    Aggregating interference constraints for Potts factor.", log_fn)
    total_indexed = sum(1 for station in graph.stations_by_id.values() if station.station_index is not None)
    seen: set[tuple[int, int, int, int]] = set()
    constraint_rows: list[tuple[int, int, int, int]] = []
    processed = 0
    for station in graph.stations_by_id.values():
        if station.station_index is None:
            continue
        processed += 1
        a_idx = station.station_index
        for interference, partner_idx in station.interferences_deduped():
            a_chan_idx = interference.subject_channel_index
            b_chan_idx = interference.other_channel_index
            key = (a_idx, partner_idx, a_chan_idx, b_chan_idx)
            if key in seen:
                raise ValueError(f"Duplicate constraint: {key}")
            seen.add(key)
            constraint_rows.append(key)
        if processed % 100 == 0:
            _emit_log(
                f"      Processed {processed:,}/{total_indexed:,} stations "
                f"(rows collected: {len(constraint_rows):,}).",
                log_fn,
            )

    if not constraint_rows:
        return None

    _emit_log(
        f"    Constraint aggregation complete: {len(constraint_rows):,} unique rows captured.",
        log_fn,
    )

    channel_count = graph.channel_count
    edge_count = len(constraint_rows)
    _emit_log(
        f"    Allocating pairwise penalty tensor: {edge_count:,} edges × "
        f"{channel_count}×{channel_count} channels.",
        log_fn,
    )
    weights = np.zeros((edge_count, channel_count, channel_count), dtype=np.float32)
    head_nodes: list[CategoricalNode] = []
    tail_nodes: list[CategoricalNode] = []

    for idx, (a_idx, b_idx, a_chan_idx, b_chan_idx) in enumerate(constraint_rows):
        head_nodes.append(nodes[a_idx])
        tail_nodes.append(nodes[b_idx])
        weights[idx, a_chan_idx, b_chan_idx] = -float(penalty)
        if (idx + 1) % 250000 == 0:
            _emit_log(f"      Populated {idx + 1:,}/{edge_count:,} pairwise entries.", log_fn)

    weights_jnp = jnp.asarray(weights)
    return CategoricalEBMFactor([Block(head_nodes), Block(tail_nodes)], weights_jnp), weights_jnp


def _prepare_initial_state(
    graph: TVGraph,
    free_blocks: Sequence[Block],
    seed: int,
    *,
    init_random: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[jnp.ndarray], int, int]:
    """
    Build the sampler's initial block state, preferring post-auction channel assignments unless
    ``init_random`` requests random initialisation within each station domain.
    """

    key = jax.random.PRNGKey(seed)
    init_state: list[jnp.ndarray] = []
    post_count = 0
    random_count = 0
    for station_index, block in enumerate(free_blocks):
        station_id = graph.station_id_for_index(station_index)
        station = graph.station(station_id)
        new_channel = station.new_channel
        if not init_random and new_channel is not None:
            if new_channel not in graph.channel_values:
                channel_idx = graph.channel_count - 1
                _emit_warning(
                    f"Station {station.station_id} has a new channel that is not in the graph channel set. Using the highest channel index, {channel_idx}.",
                    log_fn,
                )
            else:   
                channel_idx = graph.channel_index_for_channel(new_channel)
            post_count += 1
        else:
            domain = np.asarray(station.domain_indices, dtype=np.int32)
            if domain.size == 0:
                raise ValueError(f"Station {station.station_id} has an empty domain.")
            key, subkey = jax.random.split(key)
            channel_idx = int(jax.random.choice(subkey, domain))
            random_count += 1
        init_state.append(jnp.asarray([channel_idx], dtype=jnp.uint8))
    return init_state, post_count, random_count


def _build_thrml_components(
    graph: TVGraph,
    *,
    lambda_conflict: float,
    lambda_domain: float,
    seed: int,
    init_random: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> PottsComponents:
    """
    Convert the TVGraph into THRML nodes, factors, and a compiled sampling program.
    """

    nodes = [CategoricalNode() for _ in graph.station_ids_by_index]
    free_blocks = [Block([node]) for node in nodes]

    _emit_log("→ Creating THRML categorical nodes (one per station) and single-node Gibbs blocks.", log_fn)
    domain_factor = _make_domain_factor(graph, nodes, penalty=lambda_domain, log_fn=log_fn)
    _emit_log("→ Wiring domain factor: disallowed channels receive λ_domain energy penalty.", log_fn)
    conflict_factor_tuple = _make_interference_factor(graph, nodes, penalty=lambda_conflict, log_fn=log_fn)
    conflict_factor_index: int | None = None
    conflict_base_weights: jnp.ndarray | None = None
    if conflict_factor_tuple is None:
        _emit_log("→ No interference pairs detected; only unary factors are active.", log_fn)
        factors: list[CategoricalEBMFactor] = [domain_factor]
    else:
        _emit_log("→ Wiring pairwise Potts factor: λ_conflict penalises forbidden channel pairings.", log_fn)
        conflict_factor, conflict_base_weights = conflict_factor_tuple
        factors = [domain_factor, conflict_factor]
        conflict_factor_index = len(factors) - 1

    gibbs_spec = BlockGibbsSpec(
        free_super_blocks=free_blocks,
        clamped_blocks=[],
        node_shape_dtypes=DEFAULT_NODE_SHAPE_DTYPES,
    )
    samplers = [CategoricalGibbsConditional(graph.channel_count) for _ in free_blocks]
    program = FactorSamplingProgram(gibbs_spec, samplers, factors, other_interaction_groups=[])
    _emit_log("→ Compiled FactorSamplingProgram with categorical Gibbs conditionals.", log_fn)

    init_state, init_post_count, init_random_count = _prepare_initial_state(
        graph,
        free_blocks,
        seed,
        init_random=init_random,
        log_fn=log_fn,
    )
    if init_random:
        _emit_log("→ Initial state seeded randomly from station domains (post-auction assignments ignored).", log_fn)
    else:
        _emit_log("→ Initial state seeded from post-auction channels where available.", log_fn)

    _emit_log(f"→ Initial state: {init_post_count} post-auction, {init_random_count} random draws.", log_fn)

    return PottsComponents(
        program=program,
        factors=factors,
        free_blocks=free_blocks,
        init_state=init_state,
        lambda_conflict_initial=lambda_conflict,
        lambda_conflict_current=lambda_conflict,
        conflict_factor_index=conflict_factor_index,
        conflict_base_weights=conflict_base_weights,
    )


def _block_assignment_to_array(block_state: Sequence[ArrayLike]) -> np.ndarray:
    """
    Flatten a block-aligned state (list of arrays) into a 1-D NumPy vector of channel indices.
    """

    return np.asarray([int(np.asarray(block).item()) for block in block_state], dtype=np.int32)


def _set_conflict_penalty_value(components: PottsComponents, new_value: float) -> float | None:
    """
    Update the conflict factor weights to match a new λ_conflict value.
    """

    index = components.conflict_factor_index
    base_weights = components.conflict_base_weights
    initial = components.lambda_conflict_initial
    if (
        index is None
        or base_weights is None
        or initial == 0.0
    ):
        return None

    scale = float(new_value) / float(initial)
    factor = components.factors[index]
    scaled_weights = base_weights * scale

    # Try to mutate the existing factor in place; fall back to recreating it.
    updated_factor: CategoricalEBMFactor
    try:
        object.__setattr__(factor, "weights", scaled_weights)
        updated_factor = factor
    except Exception:
        updated_factor = CategoricalEBMFactor(factor.blocks, scaled_weights)
        components.factors[index] = updated_factor

    try:
        components.program.factors[index] = updated_factor  # type: ignore[attr-defined]
    except Exception:
        pass

    components.factors[index] = updated_factor
    components.lambda_conflict_current = float(new_value)
    return components.lambda_conflict_current


def _evaluate_energy(components: PottsComponents, block_state: Sequence[jnp.ndarray]) -> float:
    """
    Sum factor energies for a block state.
    """

    global_state = block_state_to_global(list(block_state), components.program.gibbs_spec)
    total = 0.0
    for factor in components.factors:
        total += float(factor.energy(global_state, components.program.gibbs_spec))
    return total


def _score_assignment(graph: TVGraph, assignment: np.ndarray) -> dict[str, object]:
    """
    Measure constraint violations for a channel assignment vector.
    """

    channel_values = graph.channel_for_channel_id
    domain_mask = graph.domain_mask
    domain_violations = 0
    for station_index, channel_index in enumerate(assignment):
        if not domain_mask[station_index, channel_index]:
            domain_violations += 1

    violations = 0
    violations_by_type: Counter[str] = Counter()
    seen_pairs: set[tuple[int, int, int, int]] = set()

    for station in graph.stations_by_id.values():
        if station.station_index is None:
            continue
        a_idx = station.station_index
        assigned_a_idx = int(assignment[a_idx])
        assigned_a_val = channel_values[assigned_a_idx]

        for interference in station.interferences:
            if assigned_a_val != interference.subject_channel:
                continue
            for partner_idx in interference.station_indices:
                assigned_b_idx = int(assignment[partner_idx])
                assigned_b_val = channel_values[assigned_b_idx]
                if assigned_b_val != interference.other_channel:
                    continue

                if a_idx <= partner_idx:
                    key = (a_idx, partner_idx, assigned_a_idx, assigned_b_idx)
                    constraint_type = interference.constraint_type
                else:
                    key = (partner_idx, a_idx, assigned_b_idx, assigned_a_idx)
                    constraint_type = interference.constraint_type

                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                violations += 1
                violations_by_type[constraint_type] += 1

    return {
        "domain_violations": domain_violations,
        "violations": violations,
        "violations_by_type": violations_by_type,
    }


def _collect_samples(components: PottsComponents, schedule: SamplingSchedule, seed: int) -> np.ndarray:
    """
    Run block Gibbs sampling and gather channel indices per station.
    """

    key = jax.random.PRNGKey(seed ^ 0xC0FFEE)
    samples_by_block = sample_states(
        key,
        components.program,
        schedule,
        init_state_free=list(components.init_state),
        state_clamp=[],
        nodes_to_sample=components.program.gibbs_spec.free_blocks,
    )
    stacked = [np.asarray(block[:, 0], dtype=np.int32) for block in samples_by_block]
    return np.stack(stacked, axis=1)


def _collect_samples_with_web_viz(
    components: PottsComponents,
    schedule: SamplingSchedule,
    seed: int,
    viz: "WebVizController",
    *,
    viz_every: int,
) -> np.ndarray:
    """
    Run block Gibbs sampling while streaming updates to the web visualiser.
    """

    if schedule.n_samples <= 0:
        return np.zeros((0, len(components.init_state)), dtype=np.int32)
    if schedule.steps_per_sample <= 0:
        raise ValueError("steps_per_sample must be positive when using web visualisation.")

    key = jax.random.PRNGKey(seed ^ 0xFEED5EED)
    state_free = [jnp.asarray(block) for block in components.init_state]
    clamp_state: list[jnp.ndarray] = []
    sampler_states = [sampler.init() for sampler in components.program.samplers]

    samples: list[np.ndarray] = []
    step = 0
    last_assignment: np.ndarray | None = None
    last_logged_step: int | None = None

    def apply_control_updates() -> None:
        updates = viz.poll_control_updates()
        for update in updates:
            if update.get("type") == "set_lambda_conflict":
                try:
                    target = float(update["value"])
                except (TypeError, ValueError):
                    continue
                applied = _set_conflict_penalty_value(components, target)
                if applied is not None:
                    viz.log(f"λ_conflict updated to {applied:.4f}", level="progress")
                    viz.broadcast_lambda_conflict(applied)

    def maybe_update(current_step: int, force: bool = False) -> np.ndarray:
        nonlocal last_assignment, last_logged_step
        assignment = _block_assignment_to_array(state_free)
        should_refresh = force or (viz_every > 0 and current_step % viz_every == 0)
        if should_refresh and last_logged_step != current_step:
            energy = _evaluate_energy(components, state_free)
            viz.record_state(current_step, assignment, energy)
            viz.log(
                f"Step {current_step:,}: energy={energy:.2f}",
                level="progress",
            )
            last_logged_step = current_step
        last_assignment = assignment
        return assignment

    maybe_update(step, force=True)
    viz.log(
        "Starting Gibbs sweeps (first update may pause for JIT compilation)…",
        level="progress",
    )
    if components.conflict_factor_index is not None:
        viz.log(
            f"Initial λ_conflict: {components.lambda_conflict_current:.4f}",
            level="progress",
        )
        viz.broadcast_lambda_conflict(components.lambda_conflict_current)
    apply_control_updates()

    for _ in range(schedule.n_warmup):
        step += 1
        key, subkey = jax.random.split(key)
        state_free, sampler_states = sample_blocks(
            subkey,
            state_free,
            clamp_state,
            components.program,
            sampler_states,
        )
        maybe_update(step)
        apply_control_updates()

    assignment = maybe_update(step, force=True)
    samples.append(assignment.copy())
    apply_control_updates()

    for _ in range(1, schedule.n_samples):
        for _ in range(schedule.steps_per_sample):
            step += 1
            key, subkey = jax.random.split(key)
            state_free, sampler_states = sample_blocks(
                subkey,
                state_free,
                clamp_state,
                components.program,
                sampler_states,
            )
            maybe_update(step)
            apply_control_updates()
        assignment = maybe_update(step, force=True)
        samples.append(assignment.copy())
        apply_control_updates()

    return np.stack(samples, axis=0)


def _summarise_samples(
    graph: TVGraph,
    components: PottsComponents,
    samples: np.ndarray,
) -> tuple[int, dict[str, object], float]:
    """
    Identify the lowest-violation sample and report its stats and energy.
    """

    best_index = 0
    best_assignment = samples[0]
    best_stats = _score_assignment(graph, best_assignment)
    best_energy = _evaluate_energy(
        components, [jnp.asarray([value], dtype=jnp.uint8) for value in best_assignment]
    )

    for idx in range(1, samples.shape[0]):
        assignment = samples[idx]
        stats = _score_assignment(graph, assignment)
        energy = _evaluate_energy(components, [jnp.asarray([value], dtype=jnp.uint8) for value in assignment])
        if stats["violations"] < best_stats["violations"] or (
            stats["violations"] == best_stats["violations"] and energy < best_energy
        ):
            best_index = idx
            best_stats = stats
            best_energy = energy

    return best_index, best_stats, best_energy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample Potts-model colourings for an FCC TVGraph using THRML.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Dataset directory or preset name relative to ./input containing the station/channel graph to solve.",
    )
    parser.add_argument(
        "-lambda-conflict",
        "--lambda-conflict",
        type=float,
        default=8.0,
        help="Energy penalty applied whenever a channel assignment violates a constraint; increase to discourage conflicts.",
    )
    parser.add_argument(
        "-lambda-domain",
        "--lambda-domain",
        type=float,
        default=100.0,
        help="Energy penalty for assigning a station to a channel outside its permitted domain; increase to keep stations in-range.",
    )
    parser.add_argument(
        "-warmup",
        "--warmup",
        type=int,
        default=0,
        help="Number of full Gibbs sweeps to discard before collecting samples; lets the Markov chain mix.",
    )
    parser.add_argument(
        "-samples",
        "--samples",
        type=int,
        default=50,
        help="Total station-to-channel assignments to retain after warmup; each stored sample is a complete solution.",
    )
    parser.add_argument(
        "-steps-per-sample",
        "--steps-per-sample",
        type=int,
        default=1,
        help="Full Gibbs sweeps to run between stored samples (thinning) to reduce autocorrelation.",
    )
    parser.add_argument(
        "-seed",
        "--seed",
        type=int,
        default=0,
        help="Random seed controlling initial state selection and all sampler RNG; set for reproducible runs.",
    )
    parser.add_argument(
        "-r",
        "--init-random",
        action="store_true",
        help="Start from a random channel assignment instead of the post-auction baseline when initialising the sampler.",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable the FastAPI+D3 web UI (enabled by default) that streams sampler progress and intermediate stats.",
    )
    parser.add_argument(
        "-web-viz-host",
        "--web-viz-host",
        default="127.0.0.1",
        help="Hostname/interface for the web visualisation server (e.g. 0.0.0.0 to listen on all addresses).",
    )
    parser.add_argument(
        "-web-viz-port",
        "--web-viz-port",
        type=int,
        default=8765,
        help="TCP port that the web visualisation server exposes.",
    )
    parser.add_argument(
        "-web-viz-history-dir",
        "--web-viz-history-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where the web visualiser writes NDJSON history logs for later inspection.",
    )
    parser.add_argument(
        "-web-viz-no-open",
        "--web-viz-no-open",
        action="store_true",
        help="Prevent automatically opening a browser tab when the web visualisation is enabled; only print the URL.",
    )
    parser.add_argument(
        "-web-viz-no-block",
        "--web-viz-no-block",
        action="store_true",
        help=(
            "If specified while the web visualisation is enabled, the program will exit immediately after sampling completes "
            "instead of keeping the web visualisation server running for further inspection. "
            "Use this if you do not want to keep the web interface open after the run."
        ),
    )
    parser.add_argument(
        "-web-viz-every",
        "--web-viz-every",
        type=int,
        default=1,
        help="Number of Gibbs sweeps between pushes to the web UI when visualisation is active; higher values update less often.",
    )
    parser.add_argument(
        "-web-viz-run-name",
        "--web-viz-run-name",
        type=str,
        default=None,
        help="Optional label shown in the web UI and stored in logs; defaults to a timestamp.",
    )
    parser.add_argument(
        "--jax-log-compiles",
        action="store_true",
        help="Enable verbose JAX compilation logging to stdout for diagnosing long compile times.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    compile_logging_enabled = False
    if args.jax_log_compiles:
        jax.config.update("jax_log_compiles", True)
        compile_logging_enabled = True

    graph = TVGraph(args.input)

    controller: WebVizController | None = None
    progress_messages: list[str] = []
    progress_replay_done = False

    def record_progress(message: str) -> None:
        print(message)
        progress_messages.append(message)
        if controller is not None and progress_replay_done:
            controller.log(message, level="progress", broadcast=True)

    def record_warning(message: str) -> None:
        record_progress(f"{WARNING_PREFIX}: {message}")

    if compile_logging_enabled:
        record_progress("JAX compile logging enabled (jax_log_compiles=True).")

    record_progress(f"Loaded graph with {graph.station_count:,} stations and {graph.channel_count:,} channels.")
    graph_stats_lines = _summarise_graph_stats(graph)
    record_progress("Graph statistics snapshot:")
    for line in graph_stats_lines:
        record_progress(f"  {line}")

    has_coordinates = any(
        station.lat is not None and station.lon is not None
        for station in graph.stations_by_id.values()
    )
    if not has_coordinates:
        record_warning(
            "Loaded dataset is missing latitude/longitude coordinates; map-based visualisations will not display station locations."
        )

    devices = jax.devices()
    devices_summary = ", ".join(str(device) for device in devices) or "none"
    record_progress(f"JAX devices: {devices_summary}")
    default_backend = jax.default_backend()
    record_progress(f"JAX default backend: {default_backend}")

    components = _build_thrml_components(
        graph,
        lambda_conflict=args.lambda_conflict,
        lambda_domain=args.lambda_domain,
        seed=args.seed,
        init_random=args.init_random,
        log_fn=record_progress,
    )

    schedule = SamplingSchedule(n_warmup=args.warmup, n_samples=args.samples, steps_per_sample=args.steps_per_sample)
    record_progress(f"→ Running Gibbs sampling: warmup={schedule.n_warmup}, samples={schedule.n_samples}.")

    initial_assignment = _block_assignment_to_array(components.init_state)
    init_energy = _evaluate_energy(components, components.init_state)
    init_stats = _score_assignment(graph, initial_assignment)
    record_progress(
        f"Initial state: energy={init_energy:.2f}, "
        f"violations={init_stats['violations']}, "
        f"domain breaches={init_stats['domain_violations']}"
    )

    if schedule.n_samples == 0:
        return 0

    try:
        if not args.no_viz:
            try:
                from lib.tv_web_viz import WebVizConfig, WebVizController  # type: ignore[redefinition]
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Web visualisation requires the fastapi and uvicorn packages. "
                    "Install them via `pip install fastapi uvicorn`, or re-run with `--no-viz` to disable the web interface."
                ) from exc

            config = WebVizConfig(
                host=args.web_viz_host,
                port=args.web_viz_port,
                history_dir=args.web_viz_history_dir,
                auto_open=not args.web_viz_no_open,
                block=not args.web_viz_no_block,
                run_name=args.web_viz_run_name,
            )
            controller = WebVizController(
                graph,
                config,
                run_metadata={
                    "lambda_conflict": args.lambda_conflict,
                    "lambda_domain": args.lambda_domain,
                },
            )
            controller.start()
            record_progress(f"→ Web visualisation running at {controller.url}")
            for line in progress_messages:
                controller.log(line, level="progress", broadcast=True)
            progress_replay_done = True

        if controller is not None:
            viz_every = max(1, args.web_viz_every)
            samples = _collect_samples_with_web_viz(
                components,
                schedule,
                seed=args.seed,
                viz=controller,
                viz_every=viz_every,
            )
        else:
            samples = _collect_samples(components, schedule, seed=args.seed)
    except Exception:
        if controller is not None:
            controller.stop()
        raise
    best_index, best_stats, best_energy = _summarise_samples(graph, components, samples)
    best_assignment = samples[best_index]

    record_progress(
        f"Best sample #{best_index}: energy={best_energy:.2f}, "
        f"violations={best_stats['violations']}, "
        f"domain breaches={best_stats['domain_violations']}"
    )

    violations_by_type = best_stats["violations_by_type"]
    if isinstance(violations_by_type, Counter) and violations_by_type:
        record_progress("Constraint hits by type:")
        for constraint_type, count in violations_by_type.most_common():
            record_progress(f"  {constraint_type:<8} {count}")

    post_channels = graph.channel_values[best_assignment]
    record_progress("Sample assignment preview (station_id → channel):")
    preview = min(10, graph.station_count)
    for station_id, channel in zip(graph.station_ids_by_index[:preview], post_channels[:preview]):
        record_progress(f"  {station_id:<8} → {channel}")
    if graph.station_count > preview:
        record_progress("  … (truncated)")

    if controller is not None:
        if controller.config.block:
            try:
                controller.block_until_interrupt()
            finally:
                controller.stop()
        else:
            controller.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

