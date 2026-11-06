"""镜头切分：复用 PySceneDetect（BSD-3-Clause，https://github.com/Breakthrough/PySceneDetect）。"""

from __future__ import annotations

from pathlib import Path

from scenedetect import ContentDetector, detect

from ..schemas import Scene


def detect_scenes(path: str | Path, threshold: float = 27.0, duration_sec: float | None = None) -> list[Scene]:
    """返回 [start, end) 的镜头列表；单镜头/全黑帧等退化输入返回整段一个镜头。"""
    path = str(path)
    raw = detect(path, ContentDetector(threshold=threshold, min_scene_len=15), backend="opencv")
    scenes = []
    for i, (start, end) in enumerate(raw):
        s, e = start.get_seconds(), end.get_seconds()
        if duration_sec is not None:
            e = min(e, duration_sec)
        if e <= s:
            continue
        scenes.append(Scene(scene_id=f"scene_{i:03d}", start_sec=round(s, 3), end_sec=round(e, 3)))
    if not scenes and duration_sec:
        scenes = [Scene(scene_id="scene_000", start_sec=0.0, end_sec=round(duration_sec, 3))]
    return scenes
