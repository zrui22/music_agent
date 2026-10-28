"""声乐教练对话程序：聊天 + 唱歌评测一体化

用法：
    python -m src.agent.chat

命令：
    /sing [歌名]      对着麦克风唱一段（回车开始，再回车结束），自动评测
    /file <路径> [歌名]  评测本地音频文件
    /song <歌名>      设定当前在练的歌（之后 /sing 免输歌名）
    /report           重新生成上一次的详细评语
    /help             帮助
    /quit             退出

直接打字则与教练自由对话（问技巧、问练声方法等）。
未配置 LLM 后端时退化为纯命令模式：只出分数表，不写评语。
"""
import json
import sys
import tempfile
import time
import wave
from pathlib import Path

from src.agent.diagnose import diagnose
from src.agent.scorer import Scorer

PERSONA = """你是一位经验丰富、鼓励为主的声乐教练「小嗓」。用户是业余唱歌爱好者。
你的职责：
1. 回答声乐技巧、练声方法、选歌建议等问题，具体可执行，不空谈；
2. 当用户发来 [评测结果] JSON 时，把它翻译成通俗易懂的中文点评：
   先一句话总评，再逐维说问题（分数低于 3.5 的维度重点说），最后给 2-3 条本周练习建议；
   objective 里的 pitch/rhythm 是与原曲对齐算出的客观分，subjective 是 AI 模型的主观评价，
   warnings 里的提示要如实转达（如匹配置信度低）；
3. 语气亲切专业，像真人教练，少用术语，用比喻；不要暴露 JSON 字段名。
分数均为 0-5 分制。"""

HELP = """命令：
  /sing [歌名]        录音演唱并评测（回车开始，回车结束）
  /file <路径> [歌名] 评测音频文件
  /song <歌名>        设定当前练习曲目
  /report             重新解读上次评测
  /quit               退出
直接输入文字即可和教练聊天。"""

SCORE_LABELS = {"pitch": "音准", "rhythm": "节奏", "stability": "气息稳定",
                "timbre": "音色", "expression": "表现力", "overall": "综合"}


def _fmt_scores(diag: dict) -> str:
    """分数表（无 LLM 时的兜底输出）。"""
    lines = []
    m = diag.get("match") or {}
    if m.get("song"):
        lines.append(f"匹配曲目：{m['song']} - {m.get('artists') or '?'} "
                     f"（置信 {m.get('confidence', 0):.2f}）")
    for group in ("objective", "subjective"):
        d = diag.get(group) or {}
        for k, v in d.items():
            if isinstance(v, (int, float)) and k in SCORE_LABELS:
                lines.append(f"  {SCORE_LABELS[k]}：{v:.2f}")
    for w in diag.get("warnings") or []:
        lines.append(f"  ⚠ {w}")
    return "\n".join(lines) if lines else "（无评分结果）"


def _record_wav() -> str | None:
    """回车开始、回车结束的麦克风录音。无录音设备时返回 None。"""
    try:
        import sounddevice as sd
    except Exception as e:  # ImportError 或 OSError(PortAudio 缺失)
        print(f"（录音不可用：{e}；可改用 /file 评测音频文件。"
              f"Linux 需系统有 PortAudio：conda install -c conda-forge portaudio）")
        return None
    try:
        sd.check_input_settings(samplerate=44100, channels=1)
    except Exception as e:
        print(f"（检测不到可用麦克风：{e}；可改用 /file 评测音频文件）")
        return None
    input("🎤 回车开始录音...")
    frames = []
    stream = sd.InputStream(samplerate=44100, channels=1, dtype="float32",
                            blocksize=4096)
    stream.start()
    print("🔴 录音中，请开始演唱（再按回车结束）")
    try:
        while True:
            import select
            if select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.readline()
                break
            block, _ = stream.read(4096)
            frames.append(block.copy())
    finally:
        stream.stop()
        stream.close()
    if not frames:
        return None
    import numpy as np
    audio = np.concatenate(frames)[:, 0]
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    path = Path(tempfile.mkdtemp(prefix="sing_")) / "take.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(pcm.tobytes())
    print(f"录音 {len(audio)/44100:.1f}s")
    return str(path)


def main():
    print("🎵 声乐教练「小嗓」已上线。输入 /help 看命令，直接打字聊天。")
    try:
        from src.agent import llm_client
        llm_ok = bool(llm_client._load_codex().get("CODEX_API_KEY")
                      or llm_client._load_anthropic().get("ANTHROPIC_API_KEY"))
    except Exception:
        llm_ok = False
    if not llm_ok:
        print("（未配置 LLM 后端：只出分数表。配置 CODEX_* 或 ANTHROPIC_* 环境变量后可获得教练评语）")

    print("加载评分模型...")
    scorer = Scorer()
    song_ctx = None          # 当前练习曲目
    last_diag = None         # 上一次评测结果
    history = []             # LLM 对话历史

    def say(user_text: str) -> str:
        if not llm_ok:
            return "（纯命令模式）" + HELP
        history.append({"role": "user", "content": user_text})
        del history[:-12]
        try:
            return llm_client.chat(history, system=PERSONA)
        except Exception as e:
            return f"（教练掉线了：{e}）"

    def run_diag(path: str, song: str | None):
        nonlocal last_diag
        t0 = time.time()
        print("评测中（首次需下载/加载模型，请稍候）...")
        try:
            diag = diagnose(path, song=song, scorer=scorer)
        except Exception as e:
            print(f"评测失败：{e}")
            return
        diag["elapsed_s"] = round(time.time() - t0, 1)
        last_diag = diag
        print(_fmt_scores(diag))
        if llm_ok:
            print("\n" + say("[评测结果] " + json.dumps(diag, ensure_ascii=False)))

    while True:
        try:
            line = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n下课啦，回去多喝水多练声！")
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            print("下课啦，回去多喝水多练声！")
            break
        if line == "/help":
            print(HELP)
            continue
        if line.startswith("/song"):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print(f"当前曲目：{song_ctx or '（未设定）'}")
            else:
                song_ctx = parts[1].strip()
                print(f"已设定当前曲目：《{song_ctx}》")
            continue
        if line.startswith("/file"):
            parts = line.split(maxsplit=2)
            if len(parts) < 2:
                print("用法：/file <音频路径> [歌名]")
                continue
            run_diag(parts[1], parts[2] if len(parts) > 2 else song_ctx)
            continue
        if line.startswith("/sing"):
            parts = line.split(maxsplit=1)
            song = parts[1].strip() if len(parts) > 1 else song_ctx
            path = _record_wav()
            if path:
                run_diag(path, song)
            continue
        if line == "/report":
            if last_diag is None:
                print("还没有评测记录，先 /sing 或 /file 唱一段吧。")
            else:
                print(say("[评测结果] 请重新详细解读这份结果："
                          + json.dumps(last_diag, ensure_ascii=False)))
            continue
        # 自由对话
        print(say(line))


if __name__ == "__main__":
    main()
