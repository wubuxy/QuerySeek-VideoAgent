"""FFmpeg must not consume the launching shell's piped command stream."""
import json
from pathlib import Path
import subprocess
import sys


def test_scene_extraction_preserves_parent_stdin(demo, tmp_path):
    payload = b'q\n-next-command\n'
    code = """
import json, sys
from pathlib import Path
from queryseek.ingest.frames import extract_scene_frames
from queryseek.schemas import Scene
frames=extract_scene_frames(sys.argv[1],Path(sys.argv[2]),
    [Scene(scene_id='test',start_sec=1.0,end_sec=2.0)],max_side=96)
print(json.dumps({'times':[f.timestamp_sec for f in frames],
                  'stdin':sys.stdin.read()}))
"""
    result = subprocess.run([sys.executable, '-c', code, str(demo['video']), str(tmp_path/'frames')],
        input=payload, capture_output=True, shell=False, timeout=60,
        cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    actual = json.loads(result.stdout)
    assert actual['stdin'] == payload.decode()
    assert len(actual['times']) == 1 and 1.5 <= actual['times'][0] < 2.0
