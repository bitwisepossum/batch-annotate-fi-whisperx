# batch-annotate-fi-whisperx

Batch transcribe Finnish audio files with WhisperX and forced alignment. Can be run offline after first model download.

## Features

- Transcribe single files or whole directories
- Finnish forced alignment with [wav2vec2-xlsr-1b-finnish-lm-v2](https://huggingface.co/Finnish-NLP/wav2vec2-xlsr-1b-finnish-lm-v2)
- VAD (Silero) for speech segmentation
- Support for initial prompt for domain-specific words
- Supports wav, mp3, m4a, flac, ogg, opus
- Can be fully offline after first run
- Summary table with duration, status, and processing time per file

## Requirements

- Python 3.12
- [WhisperX](https://github.com/m-bain/whisperX)
- [FFmpeg](https://ffmpeg.org/) (for `ffprobe`)

### Python packages

```
whisperx
rich
```

## Installation

Tested with venv on Arch Linux.

```bash
python -m venv venv
source venv/bin/activate
pip install whisperx rich
```

First run downloads models (needs internet). After that everything works offline.

For offline usage, set:

```bash
export HF_HUB_OFFLINE=1
```

You might see some connection errors or warnings when running offline (e.g. from HF Hub). These are harmless as long as models are already downloaded.

You might also see a `torchcodec` error about `libtorchcodec` failing to load. This doesn't affect anything, audio is loaded through `torchaudio`. If it bothers you, install a compatible version (e.g. `pip install torchcodec==0.7` for PyTorch 2.8).

## Usage

Single file:

```bash
python batch-annotate-fi-whisperx.py recording.wav
```

Whole directory:

```bash
python batch-annotate-fi-whisperx.py ./audio/
```

With output dir and model:

```bash
python batch-annotate-fi-whisperx.py ./audio/ -o ./transcripts/ -m large-v3
```

GPU:

```bash
python batch-annotate-fi-whisperx.py ./audio/ --device cuda
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `input` | Audio file or directory | *(required)* |
| `-o`, `--output-dir` | Output directory | Same as input |
| `-m`, `--model` | Whisper model size | `large-v3` |
| `--align-model` | Finnish alignment model | `Finnish-NLP/wav2vec2-xlsr-1b-finnish-lm-v2` |
| `-t`, `--threads` | CPU threads | `2` |
| `-p`, `--prompt` | Initial prompt for domain vocabulary | None |
| `--device` | `cuda` or `cpu` | `cpu` |

### Output

Per input file:

| File | Description |
|------|-------------|
| `.txt` | Plain text |
| `.srt` | SubRip subtitles |
| `.vtt` | WebVTT subtitles |
| `.tsv` | Tab-separated (start, end, text) |
| `.json` | Full WhisperX output with word-level timestamps |

---

## Acknowledgements

### Whisper

> Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. arXiv preprint arXiv:2212.04356. https://arxiv.org/abs/2212.04356

### WhisperX

> Bain, M., Huh, J., Han, T., & Zisserman, A. (2023). *WhisperX: Time-Accurate Speech Transcription of Long-Form Audio*. INTERSPEECH 2023. https://arxiv.org/abs/2303.00747

### Finnish Alignment Model

[Finnish-NLP/wav2vec2-xlsr-1b-finnish-lm-v2](https://huggingface.co/Finnish-NLP/wav2vec2-xlsr-1b-finnish-lm-v2) by Aapo Tanskanen ([aapot](https://huggingface.co/aapot)) and Rasmus Toivanen ([RASMUS](https://huggingface.co/RASMUS)), part of the [Finnish-NLP](https://huggingface.co/Finnish-NLP) community. Fixes alignment issues with Finnish and gives much better timestamps.

---

Development assisted by [Claude Code](https://claude.ai/claude-code).
