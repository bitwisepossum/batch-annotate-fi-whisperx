#!/usr/bin/env python3
# batch transcription script using whisperx with finnish alignment

import argparse
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table

from core import (
    find_audio_files, get_audio_duration, format_duration,
    transcribe_file, DEFAULT_ALIGN_MODEL,
)

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description='Batch WhisperX with Finnish alignment model'
    )
    parser.add_argument('input', type=Path, help='Input file or directory')
    parser.add_argument('-o', '--output-dir', type=Path, help='Output directory')
    parser.add_argument('-m', '--model', default='large-v3', help='Whisper model (default: large-v3)')
    parser.add_argument(
        '--align-model',
        default=DEFAULT_ALIGN_MODEL,
        help=f'Finnish alignment model (default: {DEFAULT_ALIGN_MODEL})'
    )
    parser.add_argument('-t', '--threads', type=int, default=2, help='CPU threads (default: 2)')
    parser.add_argument(
        '-p', '--prompt',
        default=None,
        help='Initial prompt for domain-specific vocabulary'
    )
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')

    args = parser.parse_args()

    if not args.input.exists():
        console.print(f"[red]Error: Path not found: {args.input}[/red]")
        return 1

    audio_files = find_audio_files(args.input)
    if not audio_files:
        console.print("[red]Error: No audio files found[/red]")
        return 1

    # default output to same dir as input
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Batch WhisperX with Finnish Alignment[/bold cyan]")
    console.print(f"Whisper model: {args.model}")
    console.print(f"Alignment model: {args.align_model}")
    console.print(f"Device: {args.device} | Threads: {args.threads}")
    console.print(f"Files: {len(audio_files)} | Output: {output_dir}\n")
    console.print("=" * 70)

    start_time = time.time()
    results = []

    # process each file one by one
    for idx, audio_file in enumerate(audio_files, 1):
        duration = get_audio_duration(audio_file)

        console.print(f"\n[{idx}/{len(audio_files)}] [bold]{audio_file.name}[/bold]")
        if duration > 0:
            console.print(f"Duration: {format_duration(duration)}")
        console.print("-" * 70)

        file_start = time.time()
        success = transcribe_file(
            audio_file, output_dir, args.model, args.align_model,
            args.prompt, args.device, args.threads
        )
        file_time = time.time() - file_start

        results.append({
            'file': audio_file.name,
            'success': success,
            'duration': duration,
            'time': file_time
        })

        if success:
            console.print(f"[green]✓ Complete[/green] in {format_duration(file_time)}")
        else:
            console.print(f"[red]✗ Failed[/red]")

    total_time = time.time() - start_time

    # print summary table at the end
    console.print("\n" + "=" * 70)
    table = Table(show_header=True, title="[bold]Summary[/bold]")
    table.add_column("File", style="white", width=40)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Status", justify="center", width=15)
    table.add_column("Time", justify="right", width=10)

    success_count = len([r for r in results if r['success']])

    for r in results:
        status = "[green]✓ OK[/green]" if r['success'] else "[red]✗ Failed[/red]"
        # truncate long filenames so the table looks ok
        name = r['file'] if len(r['file']) <= 38 else r['file'][:38] + "…"
        table.add_row(
            name,
            format_duration(r['duration']) if r['duration'] > 0 else "?",
            status,
            format_duration(r['time'])
        )

    console.print(table)
    console.print(f"\nSuccess: {success_count}/{len(audio_files)}")
    console.print(f"Total time: {format_duration(total_time)}")

    return 0 if success_count == len(audio_files) else 1


if __name__ == '__main__':
    exit(main())
