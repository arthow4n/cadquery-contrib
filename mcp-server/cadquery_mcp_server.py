# Copyright (c) CadQuery Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CadQuery MCP (Model Context Protocol) Server.

This module provides an MCP server that allows AI assistants like Claude
to execute CadQuery scripts and receive rendered images of 3D models.

Usage:
    Run as a standalone server:
        python -m cadquery_mcp_server

    Or use the entry point:
        cadquery-mcp

Configuration in Claude Code (~/.claude/settings.json):
    {
        "mcpServers": {
            "cadquery": {
                "command": "cadquery-mcp"
            }
        }
    }
"""

import argparse
import asyncio
import base64
import json
import sys
import os
import signal
import time
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, Tool, TextContent, ImageContent

import cadquery as cq
from cadquery import cqgi
from cadquery.occ_impl.exporters.svg import getSVG
from cadquery.occ_impl.exporters import export


class _CadQueryImageContent(ImageContent):
    """Image content with the pre-MCP-2 Python attribute name retained."""

    @property
    def mimeType(self) -> str:
        return self.mime_type


# Standard view projection directions
VIEWS = {
    "isometric": (-1.75, 1.1, 5),      # Default isometric view
    "front": (0, -1, 0),                # Looking at XZ plane from -Y
    "back": (0, 1, 0),                  # Looking at XZ plane from +Y
    "top": (0, 0, 1),                   # Looking at XY plane from +Z
    "bottom": (0, 0, -1),               # Looking at XY plane from -Z
    "left": (-1, 0, 0),                 # Looking at YZ plane from -X
    "right": (1, 0, 0),                 # Looking at YZ plane from +X
    "isometric_back": (1.75, -1.1, 5),  # Isometric from opposite corner
}

_TOOLSET = "all"
_EVALUATE_FILE_DEFAULT_VIEWS = ("isometric", "front", "top", "right")


async def list_tools() -> list[Tool]:
    """List available CadQuery tools."""
    tools = [
        Tool(
            name="render",
            description=(
                "Execute CadQuery Python code and return a rendered image of the 3D model. "
                "The code should use show_object() to output shapes, or assign the final result to 'result'. "
                "Example: result = cq.Workplane('XY').box(1, 2, 3). "
                "Returns SVG by default (works headlessly, no display required). "
                "Use 'view' to specify camera angle, or 'multi_view' to get multiple angles at once."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "CadQuery Python code to execute",
                    },
                    "view": {
                        "type": "string",
                        "description": "Camera view angle. Options: isometric (default), front, back, top, bottom, left, right, isometric_back",
                        "enum": ["isometric", "front", "back", "top", "bottom", "left", "right", "isometric_back"],
                        "default": "isometric",
                    },
                    "multi_view": {
                        "type": "boolean",
                        "description": "If true, returns multiple images from different angles (isometric, front, top, right). Useful for complex models.",
                        "default": False,
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels (default: 800)",
                        "default": 800,
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels (default: 600)",
                        "default": 600,
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Whether to show hidden lines (default: true)",
                        "default": True,
                    },
                },
                "required": ["code"],
            },
        ),
        Tool(
            name="inspect",
            description=(
                "Execute CadQuery code and return geometry information about the resulting shape, "
                "including bounding box dimensions, volume, surface area, and center of mass."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "CadQuery Python code to execute",
                    },
                },
                "required": ["code"],
            },
        ),
        Tool(
            name="get_parameters",
            description=(
                "Parse CadQuery code and extract the parameters (variables) that can be customized. "
                "Returns parameter names, types, and default values."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "CadQuery Python code to parse",
                    },
                },
                "required": ["code"],
            },
        ),
        Tool(
            name="export",
            description=(
                "Execute CadQuery code and export the result to a file. "
                "Supported formats: STEP, STL, SVG, DXF, AMF, 3MF, VRML, BREP."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "CadQuery Python code to execute",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Output filename (format determined by extension)",
                    },
                    "format": {
                        "type": "string",
                        "description": "Export format (optional, inferred from filename if not provided)",
                        "enum": ["STEP", "STL", "SVG", "DXF", "AMF", "3MF", "VRML", "BREP"],
                    },
                },
                "required": ["code", "filename"],
            },
        ),
        Tool(
            name="evaluate_file",
            description=(
                "Evaluate a CadQuery file once in a fresh worker with __file__ and sibling imports. "
                "Returns geometry validity, measurements, parameters, diagnostics and per-artifact status. "
                "Optional exports write STEP/STL from the same build. PNG views are inline by default; "
                "output_dir saves them instead and views=[] skips rendering. Partial failures preserve "
                "successful results and set isError. Relative output paths use the model directory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "timeout_seconds": {"type": "number", "exclusiveMinimum": 0, "default": 300,
                                        "description": "Total worker timeout including build, exports and rendering"},
                    "exports": {"type": "array", "default": [], "description": "Optional STEP/STL exports from the same evaluated shape; relative paths use the model directory",
                        "items": {"type": "object", "additionalProperties": False,
                            "properties": {"path": {"type": "string"}, "format": {"type": "string", "enum": ["STEP", "STL"]},
                                "tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 0.02},
                                "angular_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 0.1}},
                            "required": ["path", "format"]}},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CadQuery Python source file",
                    },
                    "views": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(VIEWS),
                        },
                        "default": list(_EVALUATE_FILE_DEFAULT_VIEWS),
                        "description": "Views to render",
                    },
                    "width": {
                        "type": "integer",
                        "default": 800,
                        "minimum": 1, "maximum": 4096,
                        "description": "Image width in pixels",
                    },
                    "height": {
                        "type": "integer",
                        "default": 600,
                        "minimum": 1, "maximum": 4096,
                        "description": "Image height in pixels",
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to show hidden lines",
                    },
                    "image_format": {
                        "type": "string",
                        "enum": ["png", "svg"],
                        "default": "png",
                        "description": "Image format for returned or saved views",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Optional directory for saved views; when set, image content is not returned and matching files are replaced",
                    },
                },
                "required": ["file_path"],
            },
        ),
    ]

    if _TOOLSET == "evaluate-file":
        return [tool for tool in tools if tool.name == "evaluate_file"]
    return tools


async def _list_tools_handler(_context, _params) -> ListToolsResult:
    """Adapt the public tool list to the low-level MCP server API."""
    return ListToolsResult(tools=await list_tools())


class ToolResponse(list):
    """List-compatible internal response with explicit protocol metadata."""

    def __init__(self, content=(), *, error=False, data=None):
        super().__init__(content)
        self.error = error
        self.data = data


def _error(message, stage="request", error_type=None):
    return ToolResponse([TextContent(type="text", text=message)], error=True,
                        data={"ok": False, "errors": [{"stage": stage, "type": error_type or stage.title() + "Error", "message": message}]})


def _extract_shape(build_result, env):
    """Explicit result wins; otherwise combine every show_object output."""
    selected = env.get("result")
    if selected is None:
        selected = [item.shape for item in build_result.results]
    shapes = []

    def collect(value):
        if isinstance(value, cq.Workplane):
            for item in value.vals():
                collect(item)
        elif isinstance(value, cq.Assembly):
            collect(value.toCompound())
        elif isinstance(value, cq.Shape):
            if not any(value.isSame(previous) for previous in shapes):
                shapes.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif value is not None:
            raise TypeError(f"Expected a CadQuery shape, Workplane or Assembly, got {type(value).__name__}")

    collect(selected)
    if not shapes:
        return None
    return shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)


def _geometry_data(shape):
    """B-rep bounds independent of cached triangulation; model units are mm."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    def measure(item):
        box = Bnd_Box()
        BRepBndLib.AddOptimal_s(item.wrapped, box, False, False)
        bounds = list(box.Get())
        return {"valid": item.isValid(), "bounds_mm": bounds,
                "size_mm": [bounds[i + 3] - bounds[i] for i in range(3)],
                "volume_mm3": item.Volume(), "surface_area_mm2": item.Area(),
                "center_of_mass_mm": list(item.Center().toTuple())}

    data = measure(shape)
    data["topology"] = {name: len(getattr(shape, method)()) for name, method in
                        (("solids", "Solids"), ("faces", "Faces"), ("edges", "Edges"), ("vertices", "Vertices"))}
    data["components"] = [measure(solid) for solid in shape.Solids()]
    return data


def _render_svg(shape, view_name: str, width: int, height: int, show_hidden: bool = True) -> str:
    """Render a shape to SVG from a specific view angle."""
    projection_dir = VIEWS.get(view_name, VIEWS["isometric"])

    opts = {
        "width": width,
        "height": height,
        "projectionDir": projection_dir,
        "showAxes": view_name == "isometric" or view_name == "isometric_back",
        "showHidden": show_hidden,
    }

    return getSVG(shape, opts=opts)


def _render_png(svg_content: str) -> bytes:
    """Rasterize generated SVG content without adding a Python dependency."""
    converter = shutil.which("magick") or shutil.which("convert")
    if converter is None:
        raise RuntimeError("PNG rendering requires ImageMagick ('magick' or 'convert') on PATH")

    try:
        result = subprocess.run(
            [converter, "svg:-", "png:-"],
            input=svg_content.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        detail = e.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ImageMagick failed to render PNG: {detail}") from e
    return result.stdout


def _render_image(
    shape,
    view_name: str,
    width: int,
    height: int,
    show_hidden: bool,
    image_format: str,
) -> tuple[bytes, str]:
    """Render one view in the requested format."""
    svg_content = _render_svg(shape, view_name, width, height, show_hidden)
    if image_format == "svg":
        return svg_content.encode("utf-8"), "image/svg+xml"
    if image_format == "png":
        return _render_png(svg_content), "image/png"
    raise ValueError(f"Unsupported image format: {image_format}")


def _geometry_summary(shape, build_time: float, data=None) -> str:
    data = _geometry_data(shape) if data is None else data
    bounds = data["bounds_mm"]
    lines = ["Geometry Information:", "  Units: mm", f"  Valid: {data['valid']}", "  Bounding Box:"]
    for i, axis in enumerate("XYZ"):
        lines.append(f"    {axis}: {bounds[i]:.4f} to {bounds[i+3]:.4f} (size: {data['size_mm'][i]:.4f})")
    lines.extend([f"  Volume: {data['volume_mm3']:.4f}", f"  Surface Area: {data['surface_area_mm2']:.4f}",
                  "  Center of Mass: (" + ", ".join(f"{v:.4f}" for v in data["center_of_mass_mm"]) + ")",
                  "  Topology:"])
    lines.extend(f"    {key.title()}: {value}" for key, value in data["topology"].items())
    lines.append(f"  Build Time: {build_time:.4f}s")
    return "\n".join(lines)


def _parameter_summary(params, heading: str = "Parameters found:") -> str:
    """Format CQGI parameter metadata for an MCP text result."""
    if not params:
        return "No parameters found in the script."

    lines = [heading]
    for name, param in params.items():
        type_name = param.varType.__name__ if param.varType else "unknown"
        lines.append(f"  {name}: {type_name} = {param.default_value}")
        if param.desc:
            lines.append(f"    Description: {param.desc}")
        if param.valid_values:
            lines.append(f"    Valid values: {param.valid_values}")
    return "\n".join(lines)


async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Handle tool calls."""

    if _TOOLSET == "evaluate-file" and name != "evaluate_file":
        return _error(f"Tool not enabled in the current toolset: {name}")

    if name == "render":
        return await _handle_render(arguments)
    elif name == "inspect":
        return await _handle_inspect(arguments)
    elif name == "get_parameters":
        return await _handle_get_parameters(arguments)
    elif name == "export":
        return await _handle_export(arguments)
    elif name == "evaluate_file":
        return await _handle_evaluate_file(arguments)
    else:
        return _error(f"Unknown tool: {name}")


async def _call_tool_handler(_context, params: CallToolRequestParams):
    """Adapt the public tool caller to the low-level MCP server API."""
    from contextlib import redirect_stdout, redirect_stderr
    from cadquery_evaluation import BoundedLog
    log = BoundedLog()
    try:
        if params.name == "evaluate_file":
            content = await call_tool(params.name, params.arguments or {})
        else:
            # Legacy handlers are synchronous internally, so this context does
            # not span an event-loop yield. File evaluations capture in workers.
            with redirect_stdout(log), redirect_stderr(log):
                content = await call_tool(params.name, params.arguments or {})
    except Exception as exc:
        content = _error(f"{type(exc).__name__}: {exc}")
    if log.getvalue():
        content.append(TextContent(type="text", text="Script output:\n" + log.getvalue()))
    return CallToolResult(content=list(content), isError=getattr(content, "error", False),
                          structuredContent=getattr(content, "data", None))



server = Server(
    "cadquery",
    on_list_tools=_list_tools_handler,
    on_call_tool=_call_tool_handler,
)


async def _handle_render(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Execute CadQuery code and return rendered SVG image(s)."""
    code = arguments["code"]
    view = arguments.get("view", "isometric")
    multi_view = arguments.get("multi_view", False)
    width = arguments.get("width", 800)
    height = arguments.get("height", 600)
    show_hidden = arguments.get("show_hidden", True)

    try:
        # Parse and execute the script using CQGI
        model = cqgi.parse(code)
        result = model.build()

        if result.exception:
            return _error(f"Execution error:\n{traceback.format_exception(type(result.exception), result.exception, result.exception.__traceback__)}")

        shape = _extract_shape(result, result.env)

        if shape is None:
            return _error("No shape produced. Use show_object(shape) or assign to 'result' variable.")

        # Get the underlying Shape object if it's a Workplane
        if hasattr(shape, "val"):
            shape = shape.val()

        if multi_view:
            # Return multiple views for complex models
            views_to_render = ["isometric", "front", "top", "right"]
            results = []

            for view_name in views_to_render:
                svg_content = _render_svg(shape, view_name, width, height, show_hidden)
                svg_data = base64.standard_b64encode(svg_content.encode("utf-8")).decode("utf-8")
                results.append(_CadQueryImageContent(type="image", data=svg_data, mimeType="image/svg+xml"))

            # Add a text description of the views
            results.insert(0, TextContent(
                type="text",
                text=f"Rendered {len(views_to_render)} views: {', '.join(views_to_render)}"
            ))
            return results
        else:
            # Single view
            svg_content = _render_svg(shape, view, width, height, show_hidden)
            svg_data = base64.standard_b64encode(svg_content.encode("utf-8")).decode("utf-8")
            return [_CadQueryImageContent(type="image", data=svg_data, mimeType="image/svg+xml")]

    except SyntaxError as e:
        return _error(f"Syntax error: {e}")
    except Exception as e:
        return _error(f"Error: {type(e).__name__}: {e}")


async def _handle_inspect(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Execute CadQuery code and return geometry information."""
    code = arguments["code"]

    try:
        model = cqgi.parse(code)
        result = model.build()

        if result.exception:
            return _error(f"Execution error: {result.exception}")

        shape = _extract_shape(result, result.env)

        if shape is None:
            return _error("No shape produced. Use show_object(shape) or assign to 'result' variable.")

        # Get the underlying Shape object if it's a Workplane
        if hasattr(shape, "val"):
            shape = shape.val()

        return [TextContent(type="text", text=_geometry_summary(shape, result.buildTime))]

    except Exception as e:
        return _error(f"Error: {type(e).__name__}: {e}")


async def _handle_get_parameters(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Parse CadQuery code and extract parameters."""
    code = arguments["code"]

    try:
        model = cqgi.parse(code)
        params = model.metadata.parameters

        if not params:
            return [TextContent(type="text", text="No parameters found in the script.")]

        return [TextContent(type="text", text=_parameter_summary(params))]

    except SyntaxError as e:
        return _error(f"Syntax error: {e}")
    except Exception as e:
        return _error(f"Error: {type(e).__name__}: {e}")


async def _handle_evaluate_file(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Run each file in a fresh process; model output never enters MCP stdout."""
    from cadquery_evaluation import validate_arguments
    try:
        args = validate_arguments(arguments)
    except (ValueError, TypeError) as exc:
        return _error(str(exc), "validation", type(exc).__name__)
    request_started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cadquery-evaluate-") as directory:
        request_path = Path(directory) / "request.json"
        response_path = Path(directory) / "response.json"
        request_path.write_text(json.dumps(args), encoding="utf-8")
        worker = Path(__file__).with_name("cadquery_evaluation.py")
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(worker), str(request_path), str(response_path),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            start_new_session=(os.name == "posix"),
        )

        async def stop():
            if process.returncode is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
            await process.wait()

        try:
            await asyncio.wait_for(process.wait(), args["timeout_seconds"])
        except asyncio.TimeoutError:
            await stop()
            return _error(f"Evaluation timed out after {args['timeout_seconds']} seconds; worker stopped. "
                          "Files written by the script before timeout may remain.", "timeout")
        except asyncio.CancelledError:
            await stop()
            raise
        if process.returncode != 0 or not response_path.is_file():
            return _error(f"Evaluation worker exited without a result (exit {process.returncode}).", "worker")
        try:
            payload = json.loads(response_path.read_text(encoding="utf-8"))
            timings = payload["data"]["timings_seconds"]
            timings["worker"] = timings.pop("total")
            timings["total"] = time.monotonic() - request_started
            payload["content"][0]["text"] += f"\nTotal request time: {timings['total']:.4f}s (including worker startup)"
            content = [TextContent(**item) if item["type"] == "text" else _CadQueryImageContent(**item)
                       for item in payload["content"]]
            return ToolResponse(content, error=payload["error"], data=payload["data"])
        except (ValueError, KeyError, TypeError) as exc:
            return _error(f"Invalid worker response: {exc}", "worker")


async def _handle_export(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Execute CadQuery code and export to file."""
    code = arguments["code"]
    filename = arguments["filename"]
    export_format = arguments.get("format")

    try:
        model = cqgi.parse(code)
        result = model.build()

        if result.exception:
            return _error(f"Execution error: {result.exception}")

        shape = _extract_shape(result, result.env)

        if shape is None:
            return _error("No shape produced. Use show_object(shape) or assign to 'result' variable.")

        # Get the underlying Shape if it's a Workplane
        if hasattr(shape, "val"):
            shape = shape.val()

        # Export
        export(shape, filename, exportType=export_format)

        return [TextContent(type="text", text=f"Exported to: {filename}")]

    except Exception as e:
        return _error(f"Error: {type(e).__name__}: {e}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def _parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run the CadQuery MCP server")
    parser.add_argument(
        "--toolset",
        choices=("all", "evaluate-file"),
        default="all",
        help="Toolset to expose (default: all)",
    )
    return parser.parse_args(args)


def run():
    """Entry point for the cadquery-mcp command."""
    global _TOOLSET
    _TOOLSET = _parse_args().toolset
    asyncio.run(main())


if __name__ == "__main__":
    run()
