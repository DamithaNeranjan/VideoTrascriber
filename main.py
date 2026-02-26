import subprocess
import sys

VIDEO = "C:\\Projects\\VideoTranscriber\\EnglishNews.mp4"
# VIDEO = "C:\\Projects\\VideoTranscriber\\SinhalaNews.mp4"

result = subprocess.run(
    [sys.executable, "transcriber.py", VIDEO],
    capture_output=True,
    text=True
)

print(result.stdout)

if result.returncode != 0:
    print("Transcriber exited abnormally (expected on Windows + CUDA)")

