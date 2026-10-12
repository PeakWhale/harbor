"""MCP server so AI assistants can use a Harbor bundle.

It speaks MCP (Model Context Protocol) over stdio and has two tools:

* ``describe_model``: input fields, target, prediction range and test metrics.
* ``predict``: one prediction with a low/high range.

Start it with ``harbor mcp <bundle>``. It needs the optional ``mcp`` package:
``pip install "peakwhale-harbor[mcp]"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from .bundle import Bundle, load_bundle
from .schema import InputError


def _problem_text(exc: InputError) -> str:
    lines = [exc.message]
    for p in exc.problems:
        field = f"{p['field']}: " if "field" in p else ""
        fields = f": {', '.join(p['fields'])}" if "fields" in p else ""
        lines.append(f"- {field}{p['error']}{fields}")
    return "\n".join(lines)


def build_server(bundle: Bundle | str | Path) -> MCPServer:
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    target = bundle.target["name"]
    pct = round(bundle.coverage * 100)
    fields = "; ".join(
        f"{f['name']} (number)"
        if f["type"] == "numeric"
        else f"{f['name']} (one of: {', '.join(f['categories'][:10])})"
        for f in bundle.features
    )

    server = MCPServer(
        name="peakwhale-harbor",
        version=__version__,
        instructions=(
            f"This server predicts '{target}' with the Harbor model '{bundle.title}'. "
            "Call describe_model to learn the input fields and how accurate the model is, then call predict. "
            f"Each prediction has a {pct}% range. Predictions are estimates, not financial or appraisal advice."
        ),
    )

    @server.tool(
        description=(
            f"Describe the Harbor model '{bundle.title}': input fields with types and allowed values, "
            "the target, the prediction range method and the test metrics."
        )
    )
    def describe_model() -> dict[str, Any]:
        return bundle.describe()

    @server.tool(
        description=(
            f"Predict '{target}' for one record. Pass `features` as an object with one key per input field: "
            f"{fields}. Use null for an unknown value. Returns prediction, low and high "
            f"(a {pct}% range) and notes about the input."
        )
    )
    def predict(features: dict[str, Any]) -> dict[str, Any]:
        try:
            result = bundle.predict_one(features)
        except InputError as exc:
            raise ToolError(_problem_text(exc)) from None
        return {**result, "target": target}

    return server


def run_stdio(bundle: Bundle | str | Path) -> None:
    build_server(bundle).run("stdio")
