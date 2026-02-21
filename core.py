# shared logic for batch whisperx transcription
# used by both CLI (batch-annotate-fi-whisperx.py) and GUI (gui.pyw)

import subprocess
from pathlib import Path

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus'}

WHISPER_MODELS = ['large-v3', 'large-v2', 'medium', 'small', 'base', 'tiny']

DEFAULT_ALIGN_MODEL = 'Finnish-NLP/wav2vec2-xlsr-1b-finnish-lm-v2'


# get audio length in seconds using ffprobe
def get_audio_duration(audio_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip())
    except:
        return 0


# format seconds into human readable string
def format_duration(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


# collect audio files from path (single file or whole directory)
def find_audio_files(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        files = []
        for ext in AUDIO_EXTENSIONS:
            files.extend(sorted(input_path.glob(f'*{ext}')))
        return files
    return []


# build the whisperx command line
def build_whisperx_cmd(audio_path, output_dir, model, align_model, prompt, device, threads):
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
        '--vad_onset', '0.2',
        '--vad_offset', '0.15',
        '--chunk_size', '10',
        '--threads', str(threads),
    ]
    if prompt:
        cmd += ['--initial_prompt', prompt]
    return cmd


# run whisperx on a single file, blocking
def transcribe_file(audio_path, output_dir, model, align_model, prompt, device, threads):
    cmd = build_whisperx_cmd(audio_path, output_dir, model, align_model, prompt, device, threads)
    result = subprocess.run(cmd)
    return result.returncode == 0


# run whisperx on a single file, streaming output line by line
# returns Popen object so caller can read output and cancel
def transcribe_file_stream(audio_path, output_dir, model, align_model, prompt, device, threads):
    cmd = build_whisperx_cmd(audio_path, output_dir, model, align_model, prompt, device, threads)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return process
