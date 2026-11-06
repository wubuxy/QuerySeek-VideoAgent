from pathlib import Path,PureWindowsPath
import os
import pytest
from queryseek.tools.paths import resolved_path_is_within


@pytest.mark.parametrize('candidate,root',[
    (r'\\?\C:\workspace\outputs\cache\file.json',r'C:\workspace\outputs'),
    (r'C:\workspace\outputs\file.json',r'\\?\C:\workspace\outputs'),
    (r'\\?\UNC\server\share\outputs\cache\file.json',r'\\server\share\outputs')])
def test_extended_windows_paths_are_equivalent_after_resolution(candidate,root):
    assert resolved_path_is_within(PureWindowsPath(candidate),PureWindowsPath(root))


@pytest.mark.parametrize('candidate',[
    r'\\?\D:\workspace\outputs\file',r'\\?\C:\workspace\outputs-evil\file',
    r'\\?\C:\workspace\outputs\..\escape',r'\\?\GLOBALROOT\Device\HarddiskVolume1\outputs\file',
    r'\\?\UNC\other\share\outputs\file'])
def test_equivalence_does_not_admit_traversal_cross_drive_or_device_escape(candidate):
    assert not resolved_path_is_within(PureWindowsPath(candidate),PureWindowsPath(r'C:\workspace\outputs'))


def test_real_filesystem_resolution_still_rejects_symlink_escape(tmp_path):
    root=tmp_path/'outputs';outside=tmp_path/'outside';root.mkdir();outside.mkdir();link=root/'link'
    try:link.symlink_to(outside,target_is_directory=True)
    except OSError:pytest.skip('Symlink privilege unavailable')
    assert not resolved_path_is_within((link/'file.json').resolve(),root.resolve())


@pytest.mark.skipif(os.name!='nt',reason='Windows-only resolve spelling')
def test_checked_output_accepts_inconsistent_resolve_prefix_and_rejects_escape(tmp_path,monkeypatch):
    import scripts.sampling_probe as sampling
    monkeypatch.setattr(sampling,'ROOT',tmp_path);destination=tmp_path/'outputs/cache/file.json'
    original=Path.resolve
    def prefixed(path,*args,**kwargs):
        resolved=original(path,*args,**kwargs)
        return Path('\\\\?\\'+str(resolved)) if path==destination else resolved
    monkeypatch.setattr(Path,'resolve',prefixed)
    assert str(sampling.checked_output(destination)).startswith('\\\\?\\')
    with pytest.raises(ValueError):sampling.checked_output(tmp_path/'outputs/../../outside')
