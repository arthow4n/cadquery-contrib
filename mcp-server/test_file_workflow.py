"""Regressions for real file evaluation and protocol results."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys

import pytest
from cadquery_mcp_server import _handle_evaluate_file, _call_tool_handler
from mcp.types import CallToolRequestParams


def evaluate(path, **kwargs):
    return asyncio.run(_handle_evaluate_file({'file_path': str(path), 'views': [], **kwargs}))


def test_file_context_fresh_imports_and_bounded_output(tmp_path, capsys):
    module = tmp_path / 'parameters.py'
    module.write_text('SIZE = 2\n')
    stamp = module.stat()
    py_compile.compile(str(module), doraise=True)
    source = tmp_path / 'part.py'
    source.write_text('''import cadquery as cq
from pathlib import Path
from parameters import SIZE
assert Path(__file__).name == 'part.py'
assert Path.cwd() == Path(__file__).parent
assert Path('parameters.py').read_text().startswith('SIZE')
print('captured marker')
print('x' * 20000)
result = cq.Workplane('XY').box(SIZE, 2, 3)
''')
    first = evaluate(source)
    assert not first.error, first[0].text
    assert first.data['geometry']['volume_mm3'] == pytest.approx(12)
    module.write_text('SIZE = 4\n')
    os.utime(module, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    second = evaluate(source)
    assert not second.error, second[0].text
    assert second.data['geometry']['volume_mm3'] == pytest.approx(24)
    assert second.data['local_module_sha256'][str(module)] == hashlib.sha256(module.read_bytes()).hexdigest()
    assert 'captured marker' in second.data['diagnostics']
    assert len(second.data['diagnostics']) < 16500
    assert 'truncated' in second.data['diagnostics']
    assert not capsys.readouterr().out


@pytest.mark.parametrize('selection', [
    "show_object(a); show_object(b)",
    "result = cq.Workplane('XY').newObject([a.val(), b.val()])",
    "result = cq.Assembly().add(a, name='a').add(b, name='b')",
    "show_object(a); result = [a, b]",
])
def test_multiple_objects_not_dropped(tmp_path, selection):
    path = tmp_path/'parts.py'
    path.write_text("import cadquery as cq\na=cq.Workplane('XY').box(1,2,3)\nb=a.translate((5,0,0))\n" + selection)
    result = evaluate(path)
    assert not result.error, result[0].text
    assert result.data['geometry']['topology']['solids'] == 2
    assert result.data['geometry']['volume_mm3'] == pytest.approx(12)


def test_protocol_error_has_source_line_and_output(tmp_path):
    path = tmp_path/'broken.py'
    path.write_text("print('before failure')\nraise ValueError('bad fit')\n")
    result = asyncio.run(_call_tool_handler(None, CallToolRequestParams(
        name='evaluate_file', arguments={'file_path':str(path), 'views':[]})))
    wire = result.model_dump(by_alias=True)
    assert wire['isError'] is True
    data = wire['structuredContent']
    error = data['errors'][0]
    assert error['stage'] == 'build' and error['type'] == 'ValueError'
    assert error['file'] == str(path) and error['line'] == 2
    assert 'before failure' in data['diagnostics']


@pytest.mark.parametrize('invalid', [
    {'views':['not_a_view']}, {'width':0}, {'height':True}, {'views':['top','top']},
    {'timeout_seconds':float('nan')}, {'image_format':'jpeg'},
    {'exports':[{'path':'part.py','format':'STEP'}]},
    {'exports':[{'path':'x.stl','format':'STL','tolerance':0}]},
])
def test_validation_precedes_execution(tmp_path, invalid):
    path=tmp_path/'part.py'
    marker=tmp_path/'executed'
    path.write_text(f"open({str(marker)!r},'w').write('oops')")
    result=evaluate(path, **invalid)
    assert result.error
    assert result.data['errors'][0]['stage']=='validation'
    assert not marker.exists()


def test_batch_exports_single_build_and_bounds_stable(tmp_path):
    path=tmp_path/'part.py'
    path.write_text("import cadquery as cq\nfrom pathlib import Path\np=Path('count')\np.write_text(p.read_text()+'x' if p.exists() else 'x')\nresult=cq.Workplane('XY').circle(5).extrude(3)\n")
    exports=[{'path':'part.step','format':'STEP'}, {'path':'part.stl','format':'STL'}]
    result=evaluate(path, exports=exports, views=['top'], image_format='svg', output_dir='renders')
    assert not result.error, result[0].text
    assert (tmp_path/'count').read_text()=='x'
    for item in result.data['exports']:
        assert item['sha256']==hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
    import cadquery as cq
    from cadquery_mcp_server import _geometry_data
    exported=cq.importers.importStep(str(tmp_path/'part.step')).val()
    assert _geometry_data(exported)['bounds_mm']==pytest.approx(result.data['geometry']['bounds_mm'],abs=1e-6)
    assert (tmp_path/'renders/part_top.svg').exists()
    assert result.data['geometry']['valid']
    assert len(result.data['geometry']['components'])==1
    assert {'build','export','render','total'} <= result.data['timings_seconds'].keys()


def test_render_failure_keeps_geometry_exports_and_other_views(tmp_path):
    # Exercise the worker directly in a disposable process so monkeypatching a
    # single renderer cannot leak into the server or other tests.
    path=tmp_path/'part.py'; path.write_text("import cadquery as cq\nresult=cq.Workplane('XY').box(2,3,4)")
    runner='''
import json, sys
import cadquery_mcp_server as server
from cadquery_evaluation import evaluate, validate_arguments
original = server._render_image
def render(shape, view, *args):
    if view == 'front': raise RuntimeError('renderer unavailable')
    return original(shape, view, *args)
server._render_image = render
args=validate_arguments({'file_path':sys.argv[1], 'views':['front','top'], 'image_format':'svg',
                         'exports':[{'path':'part.step','format':'STEP'}]})
args['_worker_directory']=sys.argv[2]
print(json.dumps(evaluate(args)))
'''
    completed=subprocess.run([sys.executable,'-c',runner,str(path),str(tmp_path)],capture_output=True,text=True,check=True)
    result=json.loads(completed.stdout)
    assert result['error']
    assert result['data']['geometry']['volume_mm3']==pytest.approx(24)
    assert result['data']['exports'][0]['ok']
    assert [v['ok'] for v in result['data']['views']]==[False,True]
    assert result['data']['errors'][0]['stage']=='render'
    assert any(c['type']=='image' for c in result['content'])


def test_timeout_stops_worker_and_next_request_works(tmp_path):
    path=tmp_path/'slow.py'
    marker=tmp_path/'finished'
    path.write_text(f"import time\ntime.sleep(30)\nopen({str(marker)!r},'w').write('bad')")
    result=evaluate(path,timeout_seconds=5)
    assert result.error and result.data['errors'][0]['stage']=='timeout'
    assert not marker.exists()
    path.write_text("import cadquery as cq\nresult=cq.Workplane('XY').box(1,1,1)")
    assert not evaluate(path).error


def test_legacy_protocol_errors_and_print_capture(capsys):
    result=asyncio.run(_call_tool_handler(None,CallToolRequestParams(name='inspect',arguments={
        'code':"print('legacy output')\nraise ValueError('broken')"})))
    assert result.model_dump(by_alias=True)['isError']
    assert any('legacy output' in item.text for item in result.content)
    assert not capsys.readouterr().out


def test_stdio_round_trip_errors_images_and_schema(tmp_path):
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.session import ClientSession
    path=tmp_path/'part.py'
    path.write_text("import cadquery as cq\nprint('model print stays off protocol')\nresult=cq.Workplane('XY').box(1,2,3)")
    async def run():
        parameters=StdioServerParameters(command=sys.executable,
            args=['-m','cadquery_mcp_server','--toolset','evaluate-file'])
        async with stdio_client(parameters) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                tools=await session.list_tools()
                schema=tools.tools[0].model_dump(by_alias=True)['inputSchema']
                assert 'exports' in schema['properties']
                assert schema['properties']['show_hidden']['default'] is False
                good=await session.call_tool('evaluate_file',{'file_path':str(path),'views':['top'],'image_format':'svg'})
                good=good.model_dump(by_alias=True)
                assert not good['isError']
                assert good['structuredContent']['geometry']['volume_mm3']==pytest.approx(6)
                assert any(item['type']=='image' for item in good['content'])
                bad=await session.call_tool('evaluate_file',{'file_path':str(tmp_path/'missing.py'),'views':[]})
                assert bad.model_dump(by_alias=True)['isError']
    asyncio.run(run())


def test_export_failure_preserves_other_export_and_existing_target(tmp_path):
    path=tmp_path/'part.py'
    path.write_text("import cadquery as cq\nresult=cq.Workplane('XY').box(1,2,3)")
    blocked=tmp_path/'blocked.stl'; blocked.mkdir()
    result=evaluate(path,exports=[{'path':'good.step','format':'STEP'}, {'path':'blocked.stl','format':'STL'}])
    assert result.error
    assert [e['ok'] for e in result.data['exports']]==[True,False]
    assert (tmp_path/'good.step').is_file() and blocked.is_dir()
    assert result.data['geometry']['valid']
    assert result.data['errors'][0]['stage']=='export'
    assert sorted(p.name for p in tmp_path.iterdir())==['blocked.stl','good.step','part.py']


def test_identical_shape_dedup_and_unsupported_selection(tmp_path):
    path=tmp_path/'part.py'
    path.write_text("import cadquery as cq\na=cq.Workplane('XY').box(1,2,3)\nshow_object(a)\nshow_object(a)")
    result=evaluate(path)
    assert not result.error
    assert result.data['geometry']['topology']['solids']==1
    path.write_text("result = 123")
    result=evaluate(path)
    assert result.error and result.data['errors'][0]['type']=='TypeError'
