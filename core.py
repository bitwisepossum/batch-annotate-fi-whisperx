# shared logic for batch whisperx transcription
# used by both CLI (batch-annotate-fi-whisperx.py) and GUI (gui.pyw)

import os
import subprocess
from datetime import datetime
from pathlib import Path

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus'}

WHISPER_MODELS = ['large-v3', 'large-v2', 'medium', 'small', 'base', 'tiny']

DEFAULT_ALIGN_MODEL = 'Finnish-NLP/wav2vec2-xlsr-1b-finnish-lm-v2'


# set or unset HF_HUB_OFFLINE env var
def set_offline_mode(offline):
    if offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
    else:
        os.environ.pop('HF_HUB_OFFLINE', None)


# get audio length in seconds using ffprobe
def get_audio_duration(audio_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip())
    except Exception:
        return 0


# format seconds into human readable string
def format_duration(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


# collect audio files from path (single file or whole directory)
def find_audio_files(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = []
        for ext in AUDIO_EXTENSIONS:
            files.extend(sorted(input_path.glob(f'*{ext}')))
        return files
    return []


# build the whisperx command line
def build_whisperx_cmd(audio_path, output_dir, model, align_model, prompt, device, threads,
                       vad_onset, vad_offset, chunk_size):
    cmd = [
        'whisperx',
        str(audio_path),
        '--model', model,
        '--language', 'fi',
        '--align_model', align_model,
        '--output_dir', str(output_dir),
        '--output_format', 'all',
        '--device', device,
        '--compute_type', 'int8',
        '--vad_method', 'silero',
        '--vad_onset', str(vad_onset),
        '--vad_offset', str(vad_offset),
        '--chunk_size', str(chunk_size),
        '--threads', str(threads),
    ]
    if prompt:
        cmd += ['--initial_prompt', prompt]
    return cmd


# run whisperx on a single file, blocking
def transcribe_file(audio_path, output_dir, model, align_model, prompt, device, threads,
                    vad_onset=0.2, vad_offset=0.15, chunk_size=10):
    cmd = build_whisperx_cmd(audio_path, output_dir, model, align_model, prompt, device, threads,
                             vad_onset, vad_offset, chunk_size)
    result = subprocess.run(cmd)
    return result.returncode == 0


# run whisperx on a single file, streaming output line by line
# returns Popen object so caller can read output and cancel
def transcribe_file_stream(audio_path, output_dir, model, align_model, prompt, device, threads,
                           vad_onset=0.2, vad_offset=0.15, chunk_size=10):
    cmd = build_whisperx_cmd(audio_path, output_dir, model, align_model, prompt, device, threads,
                             vad_onset, vad_offset, chunk_size)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return process


# get whisperx version string
def get_whisperx_version():
    try:
        result = subprocess.run(
            ['whisperx', '--version'],
            capture_output=True, text=True, timeout=5
        )
        ver = result.stdout.strip() or result.stderr.strip()
        return ver if ver else 'unknown'
    except Exception:
        return 'unknown'


# save transcription settings to a text file in the output directory
# overwrites on each run — only latest settings matter
def save_settings_log(output_dir, model, align_model, device, threads,
                      prompt, offline, num_files, input_path,
                      vad_onset, vad_offset, chunk_size):
    lines = [
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"whisperx version: {get_whisperx_version()}",
        "",
        f"Model: {model}",
        f"Alignment model: {align_model}",
        f"Device: {device}",
        f"Threads: {threads}",
        "Compute type: int8",
        "",
        "VAD method: silero",
        f"VAD onset: {vad_onset}",
        f"VAD offset: {vad_offset}",
        f"Chunk size: {chunk_size}",
        "",
        f"Prompt: {prompt or '(none)'}",
        f"Offline mode: {offline}",
        "",
        f"Input: {input_path}",
        f"Files: {num_files}",
    ]
    path = Path(output_dir) / 'transcription_settings.txt'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
