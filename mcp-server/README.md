# CadQuery MCP server

The customized branch provides `evaluate_file`: build a CadQuery file once,
inspect its geometry and parameters, optionally export STEP/STL, and render views.
Each evaluation runs in a fresh worker process. The legacy `render`, `inspect`,
`get_parameters` and `export` tools remain available in the default full toolset.

## Setup

From this checkout's `mcp-server/` directory:

```sh
uv sync --locked
uv run --locked cadquery-mcp --toolset evaluate-file
```

PNG rendering requires ImageMagick (`magick` or `convert`) on `PATH`; SVG needs
no raster converter. CadQuery and the MCP SDK are locked Python dependencies.
Run the regression suite with `uv run --locked pytest -q`; pytest is included
in the development dependency group. No printer is contacted by this server.

Configure an MCP client to launch the installed `cadquery-mcp` command with
`["--toolset", "evaluate-file"]`. Use the full path to the environment's command
when it is not on the client's PATH, or launch through `uv --directory` with the
absolute `mcp-server` directory. Omit `--toolset evaluate-file` for all five tools.
Restart an already-running server after updating: it cannot advertise new
arguments or change its loaded implementation until restarted.

## File evaluation

```json
{
  "file_path": "/absolute/project/model/jar/jar.py",
  "views": ["isometric", "front", "top", "right"],
  "output_dir": "renders/print",
  "exports": [
    {"path": "jar.step", "format": "STEP"},
    {"path": "jar.stl", "format": "STL", "tolerance": 0.02, "angular_tolerance": 0.1}
  ]
}
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `file_path` | required | Source file; relative input paths use the server launch directory. |
| `views` | isometric, front, top, right | Known view names; `[]` skips rendering. Also back, bottom, left, isometric_back. |
| `width`, `height` | 800, 600 | Image dimensions, each 1–4096 pixels. |
| `show_hidden` | false | Include hidden edges when useful for inspection. |
| `image_format` | png | `png` or `svg`. |
| `output_dir` | omitted | Save images here instead of returning inline images. |
| `exports` | [] | STEP/STL exports from the same selected geometry and placement. |
| `timeout_seconds` | 300 | Positive total worker timeout, including startup, build, exports and views. |

Each export requires `path` and `format` (`STEP` or `STL`). The file extension
must match. STL tessellation uses optional positive `tolerance` (default 0.02 mm)
and `angular_tolerance` (default 0.1 radians). Relative export/output paths resolve
against the model directory; absolute paths are accepted. Existing matching
outputs are replaced only after that individual output succeeds. A group of
exports is not a transaction: inspect each status after a partial failure.

Arguments are checked before executing the source. The server assumes model
coordinates are millimetres; it does not detect or convert a script's intended
units. It does not rearrange, scale, union or repair the selected geometry.

## Script contract

```python
from pathlib import Path
import cadquery as cq
from parameters import WIDTH  # sibling source is fresh on every evaluation

HERE = Path(__file__).resolve().parent
result = cq.Workplane("XY").box(WIDTH, 20, 5)
```

The worker defines absolute `__file__`, uses the model directory as its working
directory and prepends it to the Python import path. CQGI's `__name__` remains
`__cqgi__`; put the model at top level rather than under a `__main__` guard.
A fresh process and isolated bytecode lookup prevent stale local imports,
including same-size edits with unchanged timestamps. Worker path/global changes
do not leak into later evaluations or the MCP server.

A non-None `result` explicitly selects the output and takes precedence over
`show_object()` calls. It may be a Shape, Workplane, Assembly, or list/tuple of
those. All Workplane items are included. Without `result`, all `show_object()`
outputs are combined. Multiple shapes form a compound without boolean union;
identical top-level shape objects are deduplicated. Keep display/reference
geometry out of the selected printable result. Unsupported values produce an
error rather than silently dropping geometry.

The file is trusted Python code with the server user's permissions. Process
isolation prevents state leakage and permits cancellation; it is not a security
sandbox. A script may have its own file-writing side effects; those cannot be
rolled back after an error or timeout. Python stdout/stderr are captured up to
16 KiB and returned as diagnostics. Native/subprocess output is discarded in
file workers so it cannot corrupt MCP's standard-output transport.

## Results and failures

The MCP response contains a concise text summary and `structuredContent`:

- `ok`, `errors`: explicit success and stage/type/message/source-line/traceback.
  Failures set the MCP `isError` flag, including partial export/render failures.
- `geometry`: validity, precise B-rep bounds, sizes, volume, area, centre of mass,
  topology and per-solid measurements. Bounds do not depend on render meshes.
- `parameters`: CQGI parameters from the evaluated entry point; this does not
  automatically extract parameters from imported modules.
- `source_sha256`, `local_module_sha256`, `versions`: source and imported local
  Python-module hashes, Python/CadQuery/OCP/server versions. Arbitrary data files
  read by the script are not tracked; record those dependencies separately.
- `exports`, `views`: each requested artifact's status, successful paths and
  export hashes. Inline images are returned when no output directory is requested.
- `diagnostics`, `timings_seconds`: captured Python output and stage timings.

Geometry is measured before export/rendering and invalid geometry is flagged.
A render failure preserves geometry, parameters, successful exports and other
views. A failed output may leave an older file at that path: only a successful
status identifies a current artifact. Timeouts stop the worker (and its process
group on POSIX) and report an error; timeout/crash responses cannot promise
partial geometry results. Later evaluations can continue normally.

Export success, validity and matching bounds are not printability or physical-fit
certification. Independently check the actual STEP/STL pair and slice the final
print placement as required by the consuming project.

## Legacy tools

The full toolset also accepts inline Python through `code`:

| Tool | Additional arguments | Result |
| --- | --- | --- |
| `render` | `view`, `multi_view`, `width`, `height`, `show_hidden` | SVG image(s); hidden lines default true for compatibility. |
| `inspect` | none | Geometry summary. |
| `get_parameters` | none | CQGI parameter metadata. |
| `export` | `filename`, optional `format` | One export; format inferred from filename when omitted. |

Legacy inline execution remains in-process and does not provide file-worker
isolation or timeout guarantees. Prefer `evaluate_file` for repository work.
All tools report protocol errors explicitly. Python prints from legacy calls
are captured by the protocol handler; legacy native output is not intercepted.

## License

Apache License 2.0; see the repository [LICENSE](../LICENSE).
