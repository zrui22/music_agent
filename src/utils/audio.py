"""音频工具：统一转码"""
import subprocess
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def to_wav(src: Path, dst: Path, sr: int = 44100) -> bool:
    """转码为 44.1kHz 单声道 16bit wav"""
    cmd = [FFMPEG, "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr),
           "-sample_fmt", "s16", str(dst)]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0
