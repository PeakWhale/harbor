import asyncio
import sys

import pytest

pytest.importorskip("mcp")

from mcp import Client, StdioServerParameters  # noqa: E402

from peakwhale_harbor.mcp_server import build_server  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def test_tools_are_listed(bundle):
    async def go():
        async with Client(build_server(bundle)) as client:
            return await client.list_tools()

    names = {tool.name for tool in run(go()).tools}
    assert names == {"describe_model", "predict"}


def test_predict_tool(bundle, good_record):
    async def go():
        async with Client(build_server(bundle)) as client:
            return await client.call_tool("predict", {"features": good_record})

    result = run(go())
    assert not result.is_error
    body = result.structured_content
    assert body["low"] <= body["prediction"] <= body["high"]
    assert body["target"] == "repair_hours"


def test_predict_tool_bad_input(bundle, good_record):
    async def go():
        async with Client(build_server(bundle)) as client:
            return await client.call_tool("predict", {"features": {**good_record, "machine_type": "boat"}})

    result = run(go())
    assert result.is_error
    assert "Allowed values" in result.content[0].text


def test_describe_model_tool(bundle):
    async def go():
        async with Client(build_server(bundle)) as client:
            return await client.call_tool("describe_model", {})

    body = run(go()).structured_content
    assert body["target"]["name"] == "repair_hours"
    assert "range_coverage" in body["metrics"]


def test_harbor_mcp_command_over_stdio(bundle_dir, good_record):
    """The real `harbor mcp <bundle>` command, as an AI assistant would start it."""
    params = StdioServerParameters(command=sys.executable, args=["-m", "peakwhale_harbor", "mcp", str(bundle_dir)])

    async def go():
        async with Client(params) as client:
            return await client.call_tool("predict", {"features": good_record})

    body = run(go()).structured_content
    assert body["low"] <= body["prediction"] <= body["high"]
