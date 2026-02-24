#!/usr/bin/env python3
# tkinter gui for batch whisperx transcription

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import traceback
import time
from pathlib import Path

from core import (
    find_audio_files, get_audio_duration, format_duration,
    transcribe_file_stream, set_offline_mode, save_settings_log,
    WHISPER_MODELS, DEFAULT_ALIGN_MODEL, AUDIO_EXTENSIONS,
)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch WhisperX — Finnish Transcription")
        self.root.minsize(600, 500)

        self.process = None  # current whisperx subprocess
        self.running = False
        self.cancel_requested = False
        self.batch_start_time = None

        self.vad_onset_var = tk.DoubleVar(value=0.2)
        self.vad_offset_var = tk.DoubleVar(value=0.15)
        self.chunk_size_var = tk.IntVar(value=10)

        self._build_ui()

    def _build_ui(self):
        pad = {'padx': 6, 'pady': 3}

        # --- input/output section ---
        io_frame = ttk.LabelFrame(self.root, text="Files", padding=8)
        io_frame.pack(fill='x', padx=8, pady=(8, 4))

        ttk.Label(io_frame, text="Input:").grid(row=0, column=0, sticky='w')
        self.input_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.input_var).grid(row=0, column=1, sticky='ew', **pad)
        ttk.Button(io_frame, text="File…", width=6,
                   command=self._browse_file).grid(row=0, column=2, **pad)
        ttk.Button(io_frame, text="Dir…", width=6,
                   command=self._browse_input_dir).grid(row=0, column=3, **pad)

        ttk.Label(io_frame, text="Output:").grid(row=1, column=0, sticky='w')
        self.output_var = tk.StringVar()
        self.output_entry = ttk.Entry(io_frame, textvariable=self.output_var)
        self.output_entry.grid(row=1, column=1, sticky='ew', **pad)
        self._setup_placeholder(self.output_entry, self.output_var,
                                "Same as input directory")
        ttk.Button(io_frame, text="Dir…", width=6,
                   command=self._browse_output_dir).grid(row=1, column=2, **pad)

        io_frame.columnconfigure(1, weight=1)

        # --- settings section ---
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding=8)
        settings_frame.pack(fill='x', padx=8, pady=4)

        ttk.Label(settings_frame, text="Model:").grid(row=0, column=0, sticky='w')
        self.model_var = tk.StringVar(value='large-v3')
        ttk.OptionMenu(settings_frame, self.model_var, 'large-v3', *WHISPER_MODELS
                       ).grid(row=0, column=1, sticky='w', **pad)

        ttk.Label(settings_frame, text="Device:").grid(row=0, column=2, sticky='w')
        self.device_var = tk.StringVar(value='cpu')
        ttk.OptionMenu(settings_frame, self.device_var, 'cpu', 'cpu', 'cuda'
                        ).grid(row=0, column=3, sticky='w', **pad)

        ttk.Label(settings_frame, text="Threads:").grid(row=1, column=0, sticky='w')
        self.threads_var = tk.IntVar(value=2)
        ttk.Spinbox(settings_frame, from_=1, to=16, width=4,
                     textvariable=self.threads_var).grid(row=1, column=1, sticky='w', **pad)

        self.offline_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Offline mode",
                         variable=self.offline_var
                         ).grid(row=1, column=2, columnspan=2, sticky='w', **pad)

        ttk.Label(settings_frame, text="Align model:").grid(row=2, column=0, sticky='w')
        self.align_var = tk.StringVar(value=DEFAULT_ALIGN_MODEL)
        ttk.Entry(settings_frame, textvariable=self.align_var
                  ).grid(row=2, column=1, columnspan=3, sticky='ew', **pad)

        ttk.Label(settings_frame, text="Prompt:").grid(row=3, column=0, sticky='w')
        self.prompt_var = tk.StringVar()
        ttk.Entry(settings_frame, textvariable=self.prompt_var
                  ).grid(row=3, column=1, columnspan=3, sticky='ew', **pad)

        ttk.Label(settings_frame, text="VAD onset:").grid(row=4, column=0, sticky='w')
        ttk.Spinbox(settings_frame, from_=0.0, to=1.0, increment=0.05, width=6,
                    textvariable=self.vad_onset_var).grid(row=4, column=1, sticky='w', **pad)
        ttk.Label(settings_frame, text="VAD offset:").grid(row=4, column=2, sticky='w')
        ttk.Spinbox(settings_frame, from_=0.0, to=1.0, increment=0.05, width=6,
                    textvariable=self.vad_offset_var).grid(row=4, column=3, sticky='w', **pad)

        ttk.Label(settings_frame, text="Chunk size:").grid(row=5, column=0, sticky='w')
        ttk.Spinbox(settings_frame, from_=1, to=30, width=6,
                    textvariable=self.chunk_size_var).grid(row=5, column=1, sticky='w', **pad)

        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(3, weight=1)

        # --- start/cancel button ---
        self.start_btn = ttk.Button(self.root, text="Start", command=self._toggle_run)
        self.start_btn.pack(pady=6)

        # --- log area ---
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=4)
        log_frame.pack(fill='both', expand=True, padx=8, pady=(4, 4))

        self.log = scrolledtext.ScrolledText(log_frame, height=12, state='disabled',
                                              wrap='word', font=('Consolas', 9))
        self.log.pack(fill='both', expand=True)

        # --- progress bar ---
        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill='x', padx=8, pady=(0, 8))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                             maximum=100)
        self.progress_bar.pack(side='left', fill='x', expand=True)

        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack(side='right', padx=(8, 0))

        self.stage_label = ttk.Label(progress_frame, text="")
        self.stage_label.pack(side='right', padx=(8, 0))

        self.elapsed_label = ttk.Label(progress_frame, text="")
        self.elapsed_label.pack(side='right', padx=(8, 0))

    # --- file pickers ---

    def _browse_file(self):
        # build filter string from supported extensions
        exts = ' '.join(f'*{e}' for e in sorted(AUDIO_EXTENSIONS))
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", exts), ("All files", "*.*")]
        )
        if path:
            self.input_var.set(path)

    def _browse_input_dir(self):
        path = filedialog.askdirectory(title="Select input directory")
        if path:
            self.input_var.set(path)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_entry.configure(style='TEntry')
            self.output_var.set(path)

    # --- placeholder text for entry fields ---

    def _setup_placeholder(self, entry, var, placeholder):
        style = ttk.Style()
        style.map('Placeholder.TEntry',
                  foreground=[('focus', 'grey'), ('!focus', 'grey')])

        def _show():
            if not var.get():
                entry.configure(style='Placeholder.TEntry')
                entry.insert(0, placeholder)

        def _on_focus_in(_e):
            if entry.cget('style') == 'Placeholder.TEntry':
                entry.delete(0, 'end')
                entry.configure(style='TEntry')

        def _on_focus_out(_e):
            _show()

        entry.bind('<FocusIn>', _on_focus_in)
        entry.bind('<FocusOut>', _on_focus_out)
        _show()

    # --- logging ---

    def _log(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    # --- run/cancel ---

    def _toggle_run(self):
        if self.running:
            self._cancel()
        else:
            self._start()

    def _set_controls_state(self, state):
        # enable/disable input fields while running
        for child in self.root.winfo_children():
            if isinstance(child, ttk.LabelFrame):
                for widget in child.winfo_children():
                    try:
                        widget.configure(state=state)
                    except tk.TclError:
                        pass

    def _start(self):
        input_path = self.input_var.get().strip()
        if not input_path:
            self._log("Error: no input path selected")
            return

        input_path = Path(input_path)
        if not input_path.exists():
            self._log(f"Error: path not found: {input_path}")
            return

        audio_files = find_audio_files(input_path)
        if not audio_files:
            self._log("Error: no audio files found")
            return

        # output dir defaults to input's parent
        is_placeholder = self.output_entry.cget('style') == 'Placeholder.TEntry'
        output_str = '' if is_placeholder else self.output_var.get().strip()
        output_dir = Path(output_str) if output_str else input_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        self.running = True
        self.cancel_requested = False
        self.start_btn.configure(text="Cancel")
        self._set_controls_state('disabled')

        # clear log
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.configure(state='disabled')

        # start elapsed timer
        self.batch_start_time = time.time()
        self._tick()

        # run in background thread
        thread = threading.Thread(
            target=self._run_batch,
            args=(audio_files, output_dir),
            daemon=True,
        )
        thread.start()

    def _cancel(self):
        self.cancel_requested = True
        if self.process:
            self.process.terminate()
        self._log("--- cancelling ---")

    def _run_batch(self, audio_files, output_dir):
        try:
            self._run_batch_inner(audio_files, output_dir)
        except FileNotFoundError:
            self.root.after(0, self._log,
                            "\nError: whisperx not found. "
                            "Make sure whisperx and ffmpeg are installed and on PATH.")
        except Exception:
            self.root.after(0, self._log, f"\nError:\n{traceback.format_exc()}")
        finally:
            self.process = None
            self.root.after(0, self._finish)

    def _run_batch_inner(self, audio_files, output_dir):
        model = self.model_var.get()
        align_model = self.align_var.get()
        prompt = self.prompt_var.get().strip() or None
        device = self.device_var.get()
        threads = self.threads_var.get()
        vad_onset = self.vad_onset_var.get()
        vad_offset = self.vad_offset_var.get()
        chunk_size = self.chunk_size_var.get()
        total = len(audio_files)

        set_offline_mode(self.offline_var.get())

        save_settings_log(output_dir, model, align_model, device, threads,
                          prompt, self.offline_var.get(),
                          total, Path(self.input_var.get().strip()),
                          vad_onset, vad_offset, chunk_size)

        self.root.after(0, self._log,
                        f"Processing {total} file(s) → {output_dir}")
        self.root.after(0, self._log, "=" * 50)

        start_time = time.time()
        success_count = 0

        for idx, audio_file in enumerate(audio_files, 1):
            if self.cancel_requested:
                break

            duration = get_audio_duration(audio_file)
            dur_str = format_duration(duration) if duration > 0 else "?"

            self.root.after(0, self._log,
                            f"\n[{idx}/{total}] {audio_file.name} ({dur_str})")

            # update progress bar
            self.root.after(0, self._update_progress, idx - 1, total)

            # pulse progress bar while whisperx is working
            self.root.after(0, lambda: self.progress_bar.configure(mode='indeterminate'))
            self.root.after(0, self.progress_bar.start, 20)

            file_start = time.time()

            self.process = transcribe_file_stream(
                audio_file, output_dir, model, align_model,
                prompt, device, threads,
                vad_onset, vad_offset, chunk_size
            )

            # reset stage label for each file
            self.root.after(0, self.stage_label.configure, {'text': ''})

            # read output line by line
            for line in self.process.stdout:
                line = line.rstrip('\n')
                if line:
                    self.root.after(0, self._log, f"  {line}")
                    # parse whisperx stage from log output
                    if 'Performing voice activity detection' in line:
                        self.root.after(0, self.stage_label.configure, {'text': 'VAD'})
                    elif 'Performing transcription' in line:
                        self.root.after(0, self.stage_label.configure, {'text': 'Transcribing'})
                    elif 'Performing alignment' in line:
                        self.root.after(0, self.stage_label.configure, {'text': 'Aligning'})

            self.process.wait()
            self.root.after(0, self.progress_bar.stop)
            self.root.after(0, lambda: self.progress_bar.configure(mode='determinate'))
            file_time = time.time() - file_start

            if self.cancel_requested:
                break

            if self.process.returncode == 0:
                success_count += 1
                self.root.after(0, self._log,
                                f"✓ Done in {format_duration(file_time)}")
            else:
                self.root.after(0, self._log, "✗ Failed")

        total_time = time.time() - start_time

        # final progress
        self.root.after(0, self._update_progress, total, total)

        if self.cancel_requested:
            self.root.after(0, self._log, f"\nCancelled. {success_count}/{total} completed.")
        else:
            self.root.after(0, self._log,
                            f"\nDone — {success_count}/{total} succeeded "
                            f"in {format_duration(total_time)}")

    def _update_progress(self, current, total):
        pct = (current / total) * 100 if total > 0 else 0
        self.progress_var.set(pct)
        self.progress_label.configure(text=f"{current}/{total} files")

    # ticking elapsed timer + process alive check
    def _tick(self):
        if not self.running:
            return
        elapsed = time.time() - self.batch_start_time
        self.elapsed_label.configure(text=format_duration(elapsed))

        # check if subprocess died without producing output
        if self.process and self.process.poll() is not None and self.process.returncode != 0:
            self._log(f"  whisperx process exited unexpectedly (code {self.process.returncode})")

        self.root.after(1000, self._tick)

    def _finish(self):
        self.running = False
        self.batch_start_time = None
        self.start_btn.configure(text="Start")
        self.progress_bar.configure(mode='determinate')
        self.stage_label.configure(text="")
        self.elapsed_label.configure(text="")
        self._set_controls_state('normal')


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # .pyw has no console, so show errors in a dialog
        messagebox.showerror("Error", traceback.format_exc())
