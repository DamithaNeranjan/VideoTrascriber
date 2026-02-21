import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import ffmpeg
from faster_whisper import WhisperModel


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# Load model once at startup
model_path = resource_path("models/base")

model = WhisperModel(
    model_path,
    device="cpu",
    compute_type="int8"
)


def browse_file():
    filepath = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov")]
    )
    if filepath:
        file_label.config(text=filepath)
        app.selected_file = filepath


def transcribe_video():
    if not hasattr(app, "selected_file"):
        messagebox.showwarning("Warning", "Please select a video file first.")
        return

    progress_bar.start()

    text_area.config(state="normal")
    text_area.delete(1.0, tk.END)
    text_area.config(state="disabled")

    threading.Thread(target=process_transcription, daemon=True).start()


def process_transcription():
    try:
        VIDEO_PATH = app.selected_file
        AUDIO_PATH = "temp_audio.wav"

        # Extract audio
        ffmpeg.input(VIDEO_PATH).output(
            AUDIO_PATH, ac=1, ar=16000
        ).overwrite_output().run(quiet=True)

        segments, info = model.transcribe(
            AUDIO_PATH,
            beam_size=3,
            temperature=0.0,
            vad_filter=True
        )

        for segment in segments:
            # Update UI LIVE per segment
            app.after(0, append_text, segment.text)

        os.remove(AUDIO_PATH)

        app.after(0, finish_transcription)

    except Exception as e:
        app.after(0, show_error, str(e))


def append_text(text):
    text_area.config(state="normal")
    text_area.insert(tk.END, text + " ")
    text_area.see(tk.END)
    text_area.config(state="disabled")


def finish_transcription():
    progress_bar.stop()


def show_error(error_message):
    progress_bar.stop()
    messagebox.showerror("Error", error_message)


# ---------------- UI ---------------- #

app = tk.Tk()
app.title("Video Transcriber")
app.geometry("700x500")

browse_button = tk.Button(app, text="Browse Video", command=browse_file)
browse_button.pack(pady=10)

file_label = tk.Label(app, text="No file selected")
file_label.pack()

transcribe_button = tk.Button(app, text="Transcribe", command=transcribe_video)
transcribe_button.pack(pady=10)

progress_bar = ttk.Progressbar(app, mode="indeterminate")
progress_bar.pack(fill="x", padx=20, pady=10)

text_area = tk.Text(app, wrap="word", height=15)
text_area.pack(fill="both", expand=True, padx=20, pady=10)
text_area.config(state="disabled")

app.mainloop()
