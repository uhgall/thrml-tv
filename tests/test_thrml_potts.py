"""
Smoke tests for ``tools.thrml_potts`` helper utilities.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from tools.thrml_potts import build_thrml_potts_program, sample_uniform_initial_state
from tools.tv_graph import TVGraph


def _write_lines(path: Path, rows: list[list[str]]) -> None:
    path.write_text("\n".join(",".join(map(str, row)) for row in rows), encoding="utf-8")


def test_build_thrml_potts_program_smoke(tmp_path) -> None:
    base_input = Path("input").resolve()
    dataset_dir = base_input / "test-thrml-potts"

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=False)

    try:
        _write_lines(
            dataset_dir / "Domain.csv",
            [
                ["DOMAIN", 10, 21, 22],
                ["DOMAIN", 11, 21, 22],
                ["DOMAIN", 12, 21, 22],
            ],
        )
        _write_lines(
            dataset_dir / "parameters.csv",
            [
                ["FacID"],
                ["10"],
                ["11"],
                ["12"],
            ],
        )
        _write_lines(
            dataset_dir / "Interference_Paired.csv",
            [
                ["C0", 21, 21, 10, 11],
                ["C0", 21, 21, 11, 10, 12],
                ["C0", 21, 21, 12, 11],
            ],
        )

        graph = TVGraph("test-thrml-potts")
        artifacts = build_thrml_potts_program(graph, penalty=2.0, beta=0.5)

        assert len(artifacts.nodes) == 3
        assert artifacts.station_ids == (10, 11, 12)
        assert artifacts.edges == ((0, 1), (1, 2))
        assert artifacts.potts_factor is not None

        # domain mask strictly enforces allowed channels
        assert artifacts.domain_mask.shape == (3, 2)
        assert artifacts.domain_mask.all()

        key = jax.random.PRNGKey(0)
        init_state = sample_uniform_initial_state(key, artifacts)
        assert len(init_state) == 1
        assert init_state[0].shape == (3,)
        row_indices = jnp.arange(3, dtype=jnp.int32)
        assert jnp.all(
            jnp.take_along_axis(
                artifacts.domain_mask.astype(jnp.bool_),
                init_state[0][row_indices][:, None],
                axis=1,
            )
        )

        # energy is zero when colouring is proper
        zero_state = [jnp.array([0, 1, 0], dtype=jnp.uint8)]
        energy_zero = float(artifacts.ebm.energy(zero_state, [artifacts.node_block]))
        assert energy_zero == 0.0

        # giving nodes 0 and 1 the same colour incurs a positive penalty (β·λ)
        conflict_state = [jnp.array([0, 0, 1], dtype=jnp.uint8)]
        energy_conflict = float(artifacts.ebm.energy(conflict_state, [artifacts.node_block]))
        assert energy_conflict > 0.0

    finally:
        shutil.rmtree(dataset_dir)

