"""Utilities for rendering visualization artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

_DEFAULT_TEMPLATE = Path("templates/force_animation.html")


def render_force_layout_html(
    data_payload: Mapping[str, Any],
    output_path: str | Path,
    template_path: str | Path | None = None,
) -> None:
    """Render the force-layout animation HTML from the provided payload."""

    template_file = Path(template_path) if template_path is not None else _DEFAULT_TEMPLATE
    if not template_file.exists():
        raise FileNotFoundError(f"Force-layout template not found: {template_file}")

    template_text = template_file.read_text(encoding="utf-8")
    data_json = json.dumps(data_payload, ensure_ascii=False)
    html_content = template_text.replace("{{DATA_JSON}}", data_json)

    output_path = Path(output_path)
    output_path.write_text(html_content, encoding="utf-8")
