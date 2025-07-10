import tkinter as tk
import threading
import queue
import sounddevice as sd
import numpy as np
from transformers import pipeline
from scipy.io.wavfile import write
import io

class RealTimeTranscriber:
    def __init__(self, root):
        self.root = root
        self.root.title("リアルタイム文字起こし")

        self.text = tk.Text(root, height=20, width=80)
        self.text.pack()

        self.is_recording = False
        self.audio_queue = queue.Queue()

        # Whisperモデルの準備（初回実行時にダウンロードされます）
        # モデルサイズは必要に応じて変更可能です (e.g., "openai/whisper-base")
        self.transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=0) # GPUがある場合は device=0

        self.start_button = tk.Button(root, text="録音開始", command=self.start_recording)
        self.start_button.pack()

        self.stop_button = tk.Button(root, text="録音停止", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack()

    def start_recording(self):
        self.is_recording = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        self.transcribing_thread = threading.Thread(target=self.transcribe_audio)
        self.transcribing_thread.daemon = True
        self.transcribing_thread.start()

    def stop_recording(self):
        self.is_recording = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.text.insert(tk.END, "\n録音を停止しました。\n")

    def record_audio(self):
        samplerate = 16000  # Whisperが要求するサンプルレート
        channels = 1
        
        def callback(indata, frames, time, status):
            if status:
                print(status)
            if self.is_recording:
                self.audio_queue.put(indata.copy())

        with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
            while self.is_recording:
                sd.sleep(100)

    def transcribe_audio(self):
        samplerate = 16000
        while True:
            try:
                # 複数のオーディオチャンクをまとめて処理
                audio_chunks = []
                while not self.audio_queue.empty():
                    audio_chunks.append(self.audio_queue.get())
                
                if not audio_chunks:
                    sd.sleep(100)
                    continue

                audio_data = np.concatenate(audio_chunks, axis=0)
                
                # NumPy配列をWAV形式のバイナリデータに変換
                wav_io = io.BytesIO()
                write(wav_io, samplerate, audio_data)
                wav_io.seek(0)
                
                # パイプラインはファイルパスかバイト列を受け取ることができます
                result = self.transcriber(wav_io.read())
                
                if result and result["text"]:
                    transcribed_text = result["text"]
                    # GUIを更新
                    self.root.after(0, self.update_text, transcribed_text)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"文字起こしエラー: {e}")

    def update_text(self, text):
        self.text.insert(tk.END, text)
        self.text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeTranscriber(root)
    root.mainloop()
