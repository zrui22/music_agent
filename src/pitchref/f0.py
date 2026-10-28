"""F0 提取：praat-parselmouth 自相关法

输出 (times_s, midi, voiced)：midi 为浮点音高（69=A4=440Hz），未发声帧 midi=0/voiced=False。
mp3 需先转 wav（用 src.utils.audio.to_wav）。
"""
import numpy as np
import parselmouth


def extract_f0(wav_path: str, hop_s: float = 0.01, fmin: float = 75.0, fmax: float = 800.0,
               min_strength: float = 0.45):
    snd = parselmouth.Sound(str(wav_path))
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    pitch = snd.to_pitch_ac(time_step=hop_s, pitch_floor=fmin,
                            pitch_ceiling=fmax, voicing_threshold=min_strength)
    n = pitch.get_number_of_frames()
    times = np.array([pitch.get_time_from_frame_number(i + 1) for i in range(n)])
    hz = np.array([pitch.get_value_in_frame(i + 1) or 0.0 for i in range(n)])
    # praat ac 内部已用 voicing_threshold 做判别，hz=0 即未发声帧
    voiced = hz > 0
    midi = np.where(hz > 0, 69 + 12 * np.log2(np.maximum(hz, 1e-6) / 440.0), 0.0)
    return times, midi, voiced
