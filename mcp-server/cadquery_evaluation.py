"""Isolated file evaluation worker. No model code executes in the MCP process."""
import base64
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback


def validate_arguments(arguments):
    from cadquery_mcp_server import VIEWS, _EVALUATE_FILE_DEFAULT_VIEWS
    args = dict(arguments)
    file_path = args.get('file_path')
    if not isinstance(file_path, str) or not file_path:
        raise ValueError('File path is required.')
    args['file_path'] = str(Path(file_path).expanduser().resolve())
    root = Path(args['file_path']).parent
    views = args.get('views', list(_EVALUATE_FILE_DEFAULT_VIEWS))
    if not isinstance(views, list) or any(not isinstance(v, str) or v not in VIEWS for v in views):
        raise ValueError('views must be an array of known view names.')
    if len(set(views)) != len(views):
        raise ValueError('views must not contain duplicates.')
    args['views'] = views
    for key, default in [('width', 800), ('height', 600)]:
        value = args.get(key, default)
        if type(value) is not int or not 1 <= value <= 4096:
            raise ValueError(f'{key} must be an integer between 1 and 4096.')
        args[key] = value
    args.setdefault('image_format', 'png')
    if args['image_format'] not in ('png', 'svg'):
        raise ValueError("image_format must be 'png' or 'svg'.")
    args.setdefault('show_hidden', False)
    if type(args['show_hidden']) is not bool:
        raise ValueError('show_hidden must be boolean.')
    timeout = args.get('timeout_seconds', 300)
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout_seconds must be a finite positive number.')
    args['timeout_seconds'] = timeout
    if 'output_dir' in args:
        value = args['output_dir']
        if not isinstance(value, str) or not value:
            raise ValueError('output_dir must be a non-empty directory path.')
        args['output_dir'] = str((root / Path(value).expanduser()).resolve())
    exports = args.get('exports', [])
    if not isinstance(exports, list):
        raise ValueError('exports must be an array.')
    normalized = []
    paths = {args['file_path']}
    for entry in exports:
        if not isinstance(entry, dict) or set(entry) - {'path', 'format', 'tolerance', 'angular_tolerance'}:
            raise ValueError('Each export requires path and STEP/STL format, with optional tessellation tolerances.')
        item = dict(entry)
        if not isinstance(item.get('path'), str) or not item['path'] or item.get('format') not in ('STEP', 'STL'):
            raise ValueError('Each export requires path and STEP/STL format.')
        path = (root / Path(item['path']).expanduser()).resolve()
        suffixes = ('.step', '.stp') if item['format'] == 'STEP' else ('.stl',)
        if path.suffix.lower() not in suffixes:
            raise ValueError('Export path extension must match its format.')
        if str(path) in paths:
            raise ValueError('Export paths must be distinct and cannot replace the source.')
        paths.add(str(path))
        item['path'] = str(path)
        for key, default in [('tolerance', .02), ('angular_tolerance', .1)]:
            value = item.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive.')
            item[key] = value
        normalized.append(item)
    args['exports'] = normalized
    return args


class BoundedLog(io.TextIOBase):
    """Keep the start of Python stdout/stderr without unbounded memory use."""
    def __init__(self, limit=16384):
        self.limit = limit
        self.parts = []
        self.size = 0
        self.truncated = False

    def write(self, text):
        room = max(0, self.limit - self.size)
        if room:
            self.parts.append(text[:room])
        self.size += min(room, len(text))
        self.truncated |= len(text) > room
        return len(text)

    def flush(self):
        pass

    def getvalue(self):
        return ''.join(self.parts) + ('\n[output truncated]' if self.truncated else '')


def evaluate(args):
    """Only called in a fresh worker (or a disposable test process)."""
    import cadquery as cq
    from cadquery import cqgi
    from cadquery_mcp_server import _extract_shape, _geometry_data, _geometry_summary, _parameter_summary, _render_image

    started = time.monotonic()
    path = Path(args['file_path'])
    root = path.parent
    data = {'ok': False, 'file_path': str(path), 'units': 'mm', 'errors': [],
            'views': [], 'exports': [], 'timings_seconds': {},
            'versions': {'cadquery': cq.__version__, 'python': sys.version.split()[0],
                         'server': importlib.metadata.version('cadquery-mcp')}}
    try:
        data['versions']['ocp'] = importlib.metadata.version('cadquery-ocp')
    except importlib.metadata.PackageNotFoundError:
        pass
    content = []
    log = BoundedLog()
    stage = 'read'
    geometry_text = ''
    parameter_text = ''

    def failure(exc, current_stage, **details):
        frames = traceback.extract_tb(exc.__traceback__)
        error = {'stage': current_stage, 'type': type(exc).__name__, 'message': str(exc), **details}
        if isinstance(exc, SyntaxError):
            error.update(file=exc.filename or str(path), line=exc.lineno)
        elif frames:
            error.update(file=frames[-1].filename, line=frames[-1].lineno)
        error['traceback'] = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8192:]
        data['errors'].append(error)

    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            source = path.read_bytes()
            data['source_sha256'] = hashlib.sha256(source).hexdigest()
            stage = 'parse'
            model = cqgi.parse(source.decode('utf-8'))
            stage = 'build'
            # CQGI has no public initial-globals argument. Adapt its environment
            # builder inside this disposable worker, preserving CQGI parameters.
            original_builder, original_filename = cqgi.EnvironmentBuilder, cqgi.CQSCRIPT
            class FileEnvironment(original_builder):
                def build(self):
                    env = super().build()
                    env['__file__'] = str(path)
                    return env
            cqgi.EnvironmentBuilder, cqgi.CQSCRIPT = FileEnvironment, str(path)
            os.chdir(root)
            sys.path.insert(0, str(root))
            # Ignore even pre-existing timestamp-valid local bytecode. A fresh
            # process alone would still load stale same-size/same-mtime pyc files.
            sys.dont_write_bytecode = True
            sys.pycache_prefix = str(Path(args['_worker_directory']) / 'pycache')
            try:
                built = model.build()
            finally:
                cqgi.EnvironmentBuilder, cqgi.CQSCRIPT = original_builder, original_filename
            data['timings_seconds']['build'] = built.buildTime
            if built.exception:
                raise built.exception
            stage = 'geometry'
            shape = _extract_shape(built, built.env)
            if shape is None:
                raise ValueError("No shape produced. Use show_object(shape) or assign to 'result'.")
            geometry = _geometry_data(shape)
            data['geometry'] = geometry
            geometry_text = _geometry_summary(shape, built.buildTime, geometry)
            parameter_text = _parameter_summary(model.metadata.parameters, heading='Parameters:')
            data['parameters'] = {name: {'value': param.default_value,
                'type': param.varType.__name__ if param.varType else 'unknown',
                'description': param.desc} for name, param in model.metadata.parameters.items()}
            if not geometry['valid']:
                raise ValueError('The selected geometry is invalid; exports and rendering skipped.')
            # Collect hashes of imported local Python modules, not unrelated site packages.
            dependencies = {}
            for module in list(sys.modules.values()):
                filename = getattr(module, '__file__', None)
                if filename:
                    candidate = Path(filename).resolve()
                    if candidate.suffix == '.py' and candidate.is_relative_to(root) and candidate.is_file():
                        dependencies[str(candidate)] = hashlib.sha256(candidate.read_bytes()).hexdigest()
            data['local_module_sha256'] = dependencies
            stage = 'export'
            before = time.monotonic()
            for item in args['exports']:
                target = Path(item['path'])
                temporary = None
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=target.suffix, delete=False) as f:
                        temporary = Path(f.name)
                    cq.exporters.export(shape, str(temporary), exportType=item['format'],
                                        tolerance=item['tolerance'], angularTolerance=item['angular_tolerance'])
                    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
                    size = temporary.stat().st_size
                    temporary.replace(target)
                    data['exports'].append({**item, 'ok': True, 'sha256': digest, 'bytes': size})
                except Exception as exc:
                    data['exports'].append({**item, 'ok': False})
                    failure(exc, 'export', path=str(target))
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            data['timings_seconds']['export'] = time.monotonic() - before
            stage = 'render'
            before = time.monotonic()
            for view in args['views']:
                view_started = time.monotonic()
                try:
                    image, mime = _render_image(shape, view, args['width'], args['height'], args['show_hidden'], args['image_format'])
                    info = {'view': view, 'ok': True}
                    if 'output_dir' in args:
                        destination = Path(args['output_dir']) / f'{path.stem}_{view}.{args["image_format"]}'
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as f:
                            temp_image = Path(f.name)
                            f.write(image)
                        try:
                            temp_image.replace(destination)
                        finally:
                            temp_image.unlink(missing_ok=True)
                        info['path'] = str(destination)
                    else:
                        content.append({'type': 'image', 'data': base64.b64encode(image).decode('ascii'), 'mimeType': mime})
                    data['views'].append(info)
                except Exception as exc:
                    data['views'].append({'view': view, 'ok': False})
                    failure(exc, 'render', view=view)
                data['views'][-1]['seconds'] = time.monotonic() - view_started
            data['timings_seconds']['render'] = time.monotonic() - before
        except Exception as exc:
            failure(exc, stage)
    data['diagnostics'] = log.getvalue()
    data['timings_seconds']['total'] = time.monotonic() - started
    data['ok'] = not data['errors']
    lines = [f'File: {path}', geometry_text, parameter_text]
    if 'output_dir' in args:
        lines.append('Saved views:\n' + '\n'.join(v['path'] for v in data['views'] if v.get('ok')))
    else:
        lines.append('Rendered views: ' + ', '.join(v['view'] for v in data['views'] if v['ok']))
    if data['exports']:
        lines.append('Exports:\n' + '\n'.join(f"{e['path']}: {'saved' if e['ok'] else 'FAILED'}" for e in data['exports']))
    for error in data['errors']:
        label = {'parse': 'Syntax error' if error['type'] == 'SyntaxError' else 'Parse failed',
                 'build': 'Build failed', 'read': 'File not found' if error['type'] == 'FileNotFoundError' else 'Read failed'}.get(error['stage'], error['stage'].title() + ' failed')
        lines.append(f"{label}: {error['type']}: {error['message']}\n" + error['traceback'])
    if data['diagnostics']:
        lines.append('Script output:\n' + data['diagnostics'])
    lines.append('Timings (seconds): ' + json.dumps(data['timings_seconds']))
    content.insert(0, {'type': 'text', 'text': '\n\n'.join(line for line in lines if line)})
    return {'content': content, 'error': not data['ok'], 'data': data}


if __name__ == '__main__':
    request, response = map(Path, sys.argv[1:3])
    args = json.loads(request.read_text(encoding='utf-8'))
    args['_worker_directory'] = str(request.parent)
    response.write_text(json.dumps(evaluate(args)), encoding='utf-8')
