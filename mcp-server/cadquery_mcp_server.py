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
                "Read, build, and evaluate a CadQuery Python file once. Returns geometry information, "
                "parameters, and PNG or SVG renders for the requested views. PNG is returned by default; "
                "provide output_dir to save the rendered files there instead of returning image content."
            ),
            inputSchema={
                "type": "object",
                "properties": {
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
                        "description": "Image width in pixels",
                    },
                    "height": {
                        "type": "integer",
                        "default": 600,
                        "description": "Image height in pixels",
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "default": True,
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


def _extract_shape(build_result, env):
    """Extract the shape from a build result or environment."""
    # First try to get from show_object() calls
    if build_result.first_result is not None:
        return build_result.first_result.shape

    # Fall back to 'result' variable in environment
    if "result" in env:
        return env["result"]

    return None


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


def _geometry_summary(shape, build_time: float) -> str:
    """Return the geometry information shared by inspect and evaluate_file."""
    bb = shape.BoundingBox()
    info_lines = [
        "Geometry Information:",
        "  Bounding Box:",
        f"    X: {bb.xmin:.4f} to {bb.xmax:.4f} (size: {bb.xlen:.4f})",
        f"    Y: {bb.ymin:.4f} to {bb.ymax:.4f} (size: {bb.ylen:.4f})",
        f"    Z: {bb.zmin:.4f} to {bb.zmax:.4f} (size: {bb.zlen:.4f})",
    ]

    try:
        info_lines.append(f"  Volume: {shape.Volume():.4f}")
    except Exception:
        pass

    try:
        info_lines.append(f"  Surface Area: {shape.Area():.4f}")
    except Exception:
        pass

    try:
        center = shape.Center()
        info_lines.append(f"  Center of Mass: ({center.x:.4f}, {center.y:.4f}, {center.z:.4f})")
    except Exception:
        pass

    info_lines.append("  Topology:")
    for name, method in (
        ("Solids", shape.Solids),
        ("Faces", shape.Faces),
        ("Edges", shape.Edges),
        ("Vertices", shape.Vertices),
    ):
        try:
            info_lines.append(f"    {name}: {len(method())}")
        except Exception:
            pass

    info_lines.append(f"  Build Time: {build_time:.4f}s")
    return "\n".join(info_lines)


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
        return [TextContent(type="text", text=f"Tool not enabled in the current toolset: {name}")]

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
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _call_tool_handler(_context, params: CallToolRequestParams):
    """Adapt the public tool caller to the low-level MCP server API."""
    content = await call_tool(params.name, params.arguments or {})
    return CallToolResult(content=content)


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
            return [TextContent(
                type="text",
                text=f"Execution error:\n{traceback.format_exception(type(result.exception), result.exception, result.exception.__traceback__)}"
            )]

        shape = _extract_shape(result, result.env)

        if shape is None:
            return [TextContent(
                type="text",
                text="No shape produced. Use show_object(shape) or assign to 'result' variable."
            )]

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
        return [TextContent(type="text", text=f"Syntax error: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


async def _handle_inspect(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Execute CadQuery code and return geometry information."""
    code = arguments["code"]

    try:
        model = cqgi.parse(code)
        result = model.build()

        if result.exception:
            return [TextContent(
                type="text",
                text=f"Execution error: {result.exception}"
            )]

        shape = _extract_shape(result, result.env)

        if shape is None:
            return [TextContent(
                type="text",
                text="No shape produced. Use show_object(shape) or assign to 'result' variable."
            )]

        # Get the underlying Shape object if it's a Workplane
        if hasattr(shape, "val"):
            shape = shape.val()

        return [TextContent(type="text", text=_geometry_summary(shape, result.buildTime))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


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
        return [TextContent(type="text", text=f"Syntax error: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


async def _handle_evaluate_file(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Build a CadQuery file once and return its geometry, parameters, and views."""
    file_path = arguments.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return [TextContent(type="text", text="File path is required.")]

    try:
        with open(file_path, encoding="utf-8") as source_file:
            code = source_file.read()
    except FileNotFoundError:
        return [TextContent(type="text", text=f"File not found: {file_path}")]
    except (OSError, UnicodeError) as e:
        return [TextContent(type="text", text=f"Unable to read file '{file_path}': {e}")]

    try:
        model = cqgi.parse(code)
    except SyntaxError as e:
        return [TextContent(type="text", text=f"Syntax error in '{file_path}': {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Unable to parse '{file_path}': {type(e).__name__}: {e}")]

    try:
        result = model.build()
    except Exception as e:
        return [TextContent(type="text", text=f"Build failed for '{file_path}': {type(e).__name__}: {e}")]

    if result.exception:
        return [TextContent(type="text", text=f"Build failed for '{file_path}': {result.exception}")]

    shape = _extract_shape(result, getattr(result, "env", {}))
    if shape is None:
        return [TextContent(
            type="text",
            text=f"No shape produced by '{file_path}'. Use show_object(shape) or assign to 'result'.",
        )]

    if hasattr(shape, "val"):
        shape = shape.val()

    views = arguments.get("views", list(_EVALUATE_FILE_DEFAULT_VIEWS))
    if views is None:
        views = []
    if not isinstance(views, list) or not all(isinstance(view, str) for view in views):
        return [TextContent(type="text", text="views must be an array of view names.")]

    image_format = arguments.get("image_format", "png")
    if image_format not in ("png", "svg"):
        return [TextContent(type="text", text="image_format must be 'png' or 'svg'.")]

    output_dir = arguments.get("output_dir")
    if output_dir is not None and (not isinstance(output_dir, str) or not output_dir):
        return [TextContent(type="text", text="output_dir must be a non-empty directory path.")]

    rendered_views = []
    try:
        for view_name in views:
            image_data, mime_type = _render_image(
                shape,
                view_name,
                arguments.get("width", 800),
                arguments.get("height", 600),
                arguments.get("show_hidden", True),
                image_format,
            )
            rendered_views.append((view_name, image_data, mime_type))
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Unable to render view '{view_name}' for '{file_path}': {type(e).__name__}: {e}",
        )]

    summary = "\n\n".join((
        f"File: {file_path}",
        _geometry_summary(shape, result.buildTime),
        _parameter_summary(model.metadata.parameters, heading="Parameters:"),
        f"Rendered views ({image_format}): {', '.join(views) if views else 'none'}",
    ))

    if output_dir is not None:
        output_path = Path(output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            saved_paths = []
            stem = Path(file_path).stem or "model"
            for view_name, image_data, _mime_type in rendered_views:
                saved_path = output_path / f"{stem}_{view_name}.{image_format}"
                saved_path.write_bytes(image_data)
                saved_paths.append(str(saved_path))
        except OSError as e:
            return [TextContent(type="text", text=f"Unable to save views to '{output_dir}': {e}")]

        saved_summary = "Saved views:\n" + (
            "\n".join(f"  {path}" for path in saved_paths) if saved_paths else "  none"
        )
        return [TextContent(type="text", text=f"{summary}\n\n{saved_summary}")]

    results: list[TextContent | ImageContent] = [TextContent(type="text", text=summary)]
    for _view_name, image_data, mime_type in rendered_views:
        image_data_base64 = base64.standard_b64encode(image_data).decode("utf-8")
        results.append(_CadQueryImageContent(type="image", data=image_data_base64, mimeType=mime_type))

    return results


async def _handle_export(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Execute CadQuery code and export to file."""
    code = arguments["code"]
    filename = arguments["filename"]
    export_format = arguments.get("format")

    try:
        model = cqgi.parse(code)
        result = model.build()

        if result.exception:
            return [TextContent(
                type="text",
                text=f"Execution error: {result.exception}"
            )]

        shape = _extract_shape(result, result.env)

        if shape is None:
            return [TextContent(
                type="text",
                text="No shape produced. Use show_object(shape) or assign to 'result' variable."
            )]

        # Get the underlying Shape if it's a Workplane
        if hasattr(shape, "val"):
            shape = shape.val()

        # Export
        export(shape, filename, exportType=export_format)

        return [TextContent(type="text", text=f"Exported to: {filename}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


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
