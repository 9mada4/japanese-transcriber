import sys
import threading
import queue
import pyaudio
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget
from PyQt6.QtCore import QThread, pyqtSignal
from faster_whisper import WhisperModel

# モデルサイズは、"tiny", "base", "small", "medium", "large-v3" から選択できます。
# "large-v3" は高精度ですが、より多くのVRAMを必要とします。
# "medium" や "small" は、性能とリソースのバランスが良い選択肢です。
MODEL_SIZE = "large-v3"
# 日本語を指定
LANGUAGE = "ja"

class TranscriberWorker(QThread):
    """文字起こしを行うワーカースレッド"""
    transcribed_text = pyqtSignal(str)

    def __init__(self, audio_queue):
        super().__init__()
        self.audio_queue = audio_queue
        self.is_running = True
        print("Whisperモデルをロード中...")
        self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        print("モデルのロードが完了しました。")

    def run(self):
        while self.is_running:
            try:
                # キューからオーディオデータを取得
                audio_data = self.audio_queue.get(timeout=1)
                
                # 複数のチャンクを結合
                full_audio = np.concatenate(audio_data)
                
                # 文字起こし実行
                segments, _ = self.model.transcribe(full_audio, language=LANGUAGE)
                
                transcription = "".join(segment.text for segment in segments)
                if transcription.strip():
                    self.transcribed_text.emit(transcription)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"文字起こしエラー: {e}")

    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("リアルタイム文字起こし")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.layout.addWidget(self.text_edit)

        self.start_button = QPushButton("録音開始")
        self.start_button.clicked.connect(self.start_recording)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("録音停止")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button)

        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.recording_thread = None
        self.transcriber_worker = None

    def start_recording(self):
        self.is_recording = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.text_edit.clear()
        self.text_edit.append("録音を開始しました...")

        # オーディオストリームを開始
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=16000,
                                  input=True,
                                  frames_per_buffer=1024,
                                  stream_callback=self.audio_callback)
        self.stream.start_stream()

        # 録音と文字起こしのスレッドを開始
        self.recording_thread = threading.Thread(target=self.process_audio_chunks)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        self.transcriber_worker = TranscriberWorker(self.audio_queue)
        self.transcriber_worker.transcribed_text.connect(self.update_text)
        self.transcriber_worker.start()


    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.transcriber_worker:
            self.transcriber_worker.stop()
            self.transcriber_worker.wait() # スレッドが終了するのを待つ

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.text_edit.append("\n録音を停止しました。")

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            # PyAudioからのバイトデータをNumpy配列に変換
            audio_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_queue.put([audio_np])
        return (in_data, pyaudio.paContinue)

    def process_audio_chunks(self):
        """
        マイクからの音声を一定期間（例：5秒）ごとにまとめてキューに入れる
        """
        audio_buffer = []
        while self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.extend(chunk)

                # 5秒分のデータが溜まったら文字起こしキューへ
                if len(audio_buffer) * 1024 / 16000 > 5.0:
                    self.transcriber_worker.audio_queue.put(audio_buffer)
                    audio_buffer = []
            except queue.Empty:
                continue
        
        # 残りのバッファを処理
        if audio_buffer:
            self.transcriber_worker.audio_queue.put(audio_buffer)


    def update_text(self, text):
        self.text_edit.append(text)
        self.text_edit.verticalScrollBar().setValue(self.text_edit.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self.stop_recording()
        self.p.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
