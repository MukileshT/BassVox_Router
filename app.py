import os
import sys
import time
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, sosfilt, sosfilt_zi

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QSlider,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QWidget,
)


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


@dataclass
class DeviceChoice:
    index: int
    name: str


class AudioSplitterDSP:
    """Deterministic DSP split using filtering and channel operations (no AI)."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

        # Bass: low frequencies
        self.bass_sos = butter(4, 180, btype="lowpass", fs=sample_rate, output="sos")

        # Vocal-ish: center-mid and speech/singing band
        self.vocal_sos = butter(4, [220, 5000], btype="bandpass", fs=sample_rate, output="sos")

        self.bass_zi = None
        self.vocal_zi = None

    def _ensure_state(self):
        if self.bass_zi is None:
            self.bass_zi = np.tile(sosfilt_zi(self.bass_sos), (1, 1))
            self.vocal_zi = np.tile(sosfilt_zi(self.vocal_sos), (1, 1))

    def process_block(self, block: np.ndarray):
        """
        Input block shape: (N, channels)
        Output bass/vocal shape: (N, 2)
        """
        if block.ndim == 1:
            block = block[:, None]

        if block.shape[1] == 1:
            left = block[:, 0]
            right = block[:, 0]
        else:
            left = block[:, 0]
            right = block[:, 1]

        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)

        self._ensure_state()

        bass_mono, self.bass_zi = sosfilt(self.bass_sos, mid, zi=self.bass_zi)

        center_emphasis = mid - 0.25 * side
        vocal_mono, self.vocal_zi = sosfilt(self.vocal_sos, center_emphasis, zi=self.vocal_zi)

        bass = np.column_stack([bass_mono, bass_mono]).astype(np.float32)
        vocal = np.column_stack([vocal_mono, vocal_mono]).astype(np.float32)

        np.clip(bass, -1.0, 1.0, out=bass)
        np.clip(vocal, -1.0, 1.0, out=vocal)

        return bass, vocal


class SplitPlaybackEngine:
    def __init__(self, status_cb: Optional[Callable[[str], None]] = None):
        self.status_cb = status_cb or (lambda msg: None)
        self.blocksize = 2048
        self.queue_size = 32
        self.stream_latency_sec = 0.08
        self.bass_gain = 1.0
        self.vocal_gain = 1.0

        self.bass_device: Optional[int] = None
        self.vocal_device: Optional[int] = None

        self.bass_stream = None
        self.vocal_stream = None
        self.file: Optional[sf.SoundFile] = None
        self.dsp: Optional[AudioSplitterDSP] = None

        self.bass_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=self.queue_size)
        self.vocal_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=self.queue_size)

        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.file_done_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.file_lock = threading.Lock()

    def _log(self, msg: str):
        self.status_cb(msg)

    def _clear_queues(self):
        while not self.bass_queue.empty():
            self.bass_queue.get_nowait()
        while not self.vocal_queue.empty():
            self.vocal_queue.get_nowait()

    def set_devices(self, bass_device: int, vocal_device: int):
        self.bass_device = bass_device
        self.vocal_device = vocal_device

    def set_tuning(self, blocksize: int, latency_ms: int):
        self.blocksize = max(128, int(blocksize))
        self.stream_latency_sec = max(0.005, float(latency_ms) / 1000.0)

    def set_gains(self, bass_gain: float, vocal_gain: float):
        self.bass_gain = max(0.0, float(bass_gain))
        self.vocal_gain = max(0.0, float(vocal_gain))

    def load_file(self, path: str):
        self.stop()
        loaded_file = sf.SoundFile(path, mode="r")
        self.file = loaded_file
        self.dsp = AudioSplitterDSP(loaded_file.samplerate)
        self.file_done_event.clear()
        self._clear_queues()
        self._log(f"Loaded: {os.path.basename(path)} | {loaded_file.samplerate} Hz")

    def get_duration_seconds(self) -> float:
        if self.file is None:
            return 0.0
        if self.file.samplerate <= 0:
            return 0.0
        return float(self.file.frames) / float(self.file.samplerate)

    def get_position_seconds(self) -> float:
        if self.file is None:
            return 0.0
        with self.file_lock:
            pos = self.file.tell()
            sr = self.file.samplerate
        if sr <= 0:
            return 0.0
        return float(pos) / float(sr)

    def seek_seconds(self, seconds: float):
        if self.file is None:
            return
        target = max(0.0, float(seconds))
        with self.file_lock:
            samplerate = self.file.samplerate
            frame_target = int(target * samplerate)
            frame_target = max(0, min(frame_target, max(0, self.file.frames - 1)))
            self.file.seek(frame_target)
            self.file_done_event.clear()

        self._clear_queues()
        if self.file is not None:
            self.dsp = AudioSplitterDSP(self.file.samplerate)

    def _bass_callback(self, outdata, frames, time_info, status):
        if status:
            self._log(f"Bass stream status: {status}")
        outdata[:] = self._dequeue_audio(self.bass_queue, frames)

    def _vocal_callback(self, outdata, frames, time_info, status):
        if status:
            self._log(f"Vocal stream status: {status}")
        outdata[:] = self._dequeue_audio(self.vocal_queue, frames)

    @staticmethod
    def _dequeue_audio(q: "queue.Queue[np.ndarray]", frames: int):
        try:
            block = q.get_nowait()
        except queue.Empty:
            return np.zeros((frames, 2), dtype=np.float32)

        if block.shape[0] == frames:
            return block

        if block.shape[0] > frames:
            return block[:frames]

        pad = np.zeros((frames - block.shape[0], 2), dtype=np.float32)
        return np.vstack([block, pad])

    def _open_streams(self):
        if self.file is None:
            raise RuntimeError("No file loaded")
        if self.bass_device is None or self.vocal_device is None:
            raise RuntimeError("Output devices are not selected")

        loaded_file = self.file

        self.bass_stream = sd.OutputStream(
            device=self.bass_device,
            samplerate=loaded_file.samplerate,
            channels=2,
            dtype="float32",
            blocksize=self.blocksize,
            latency=self.stream_latency_sec,
            callback=self._bass_callback,
        )
        self.vocal_stream = sd.OutputStream(
            device=self.vocal_device,
            samplerate=loaded_file.samplerate,
            channels=2,
            dtype="float32",
            blocksize=self.blocksize,
            latency=self.stream_latency_sec,
            callback=self._vocal_callback,
        )

        self.bass_stream.start()
        self.vocal_stream.start()

    def _close_streams(self):
        for stream in (self.bass_stream, self.vocal_stream):
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
        self.bass_stream = None
        self.vocal_stream = None

    def _worker_loop(self):
        assert self.file is not None
        assert self.dsp is not None

        self.file_done_event.clear()

        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.03)
                continue

            with self.file_lock:
                block = self.file.read(self.blocksize, dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                self.file_done_event.set()
                self._log("Song ended.")
                break

            bass, vocal = self.dsp.process_block(block)
            bass *= self.bass_gain
            vocal *= self.vocal_gain
            np.clip(bass, -1.0, 1.0, out=bass)
            np.clip(vocal, -1.0, 1.0, out=vocal)

            try:
                self.bass_queue.put(bass, timeout=0.2)
                self.vocal_queue.put(vocal, timeout=0.2)
            except queue.Full:
                pass

        self._log("Playback worker stopped.")

    def play(self):
        if self.file is None:
            raise RuntimeError("No file loaded")

        if self.worker and self.worker.is_alive():
            self.pause_event.clear()
            self._log("Resumed.")
            return

        self.stop_event.clear()
        self.pause_event.clear()
        self.file_done_event.clear()
        self._clear_queues()

        self._open_streams()

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        self._log("Playing.")

    def pause(self):
        self.pause_event.set()
        self._log("Paused.")

    def stop(self):
        self.stop_event.set()
        self.pause_event.clear()

        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1.2)

        self._clear_queues()
        self._close_streams()

        if self.file is not None:
            self.file.seek(0)

    def close(self):
        self.stop()
        if self.file is not None:
            self.file.close()
            self.file = None


class LiveSplitEngine:
    def __init__(self, status_cb: Optional[Callable[[str], None]] = None):
        self.status_cb = status_cb or (lambda msg: None)

        self.input_stream = None
        self.bass_stream = None
        self.vocal_stream = None

        self.input_device: Optional[int] = None
        self.bass_device: Optional[int] = None
        self.vocal_device: Optional[int] = None

        self.blocksize = 1024
        self.sample_rate = 48000
        self.stream_latency_sec = 0.06
        self.bass_gain = 1.0
        self.vocal_gain = 1.0
        self.dsp = AudioSplitterDSP(self.sample_rate)

        self.bass_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=24)
        self.vocal_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=24)

    def _log(self, msg: str):
        self.status_cb(msg)

    def configure(self, input_device: int, bass_device: int, vocal_device: int, sample_rate: int):
        self.input_device = input_device
        self.bass_device = bass_device
        self.vocal_device = vocal_device
        self.sample_rate = sample_rate
        self.dsp = AudioSplitterDSP(self.sample_rate)

    def set_tuning(self, blocksize: int, latency_ms: int):
        self.blocksize = max(128, int(blocksize))
        self.stream_latency_sec = max(0.005, float(latency_ms) / 1000.0)

    def set_gains(self, bass_gain: float, vocal_gain: float):
        self.bass_gain = max(0.0, float(bass_gain))
        self.vocal_gain = max(0.0, float(vocal_gain))

    def _input_callback(self, indata, frames, time_info, status):
        if status:
            self._log(f"Live input status: {status}")

        bass, vocal = self.dsp.process_block(indata.copy())
        bass *= self.bass_gain
        vocal *= self.vocal_gain
        np.clip(bass, -1.0, 1.0, out=bass)
        np.clip(vocal, -1.0, 1.0, out=vocal)

        for q, data in ((self.bass_queue, bass), (self.vocal_queue, vocal)):
            try:
                q.put_nowait(data)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass

    @staticmethod
    def _output_callback(queue_obj):
        def callback(outdata, frames, time_info, status):
            if status:
                pass
            try:
                data = queue_obj.get_nowait()
            except queue.Empty:
                data = np.zeros((frames, 2), dtype=np.float32)

            if data.shape[0] < frames:
                pad = np.zeros((frames - data.shape[0], 2), dtype=np.float32)
                data = np.vstack([data, pad])
            elif data.shape[0] > frames:
                data = data[:frames]

            outdata[:] = data

        return callback

    def start(self):
        if None in (self.input_device, self.bass_device, self.vocal_device):
            raise RuntimeError("Input/Bass/Vocal device not configured")

        self.stop()

        self.input_stream = sd.InputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            channels=2,
            dtype="float32",
            blocksize=self.blocksize,
            latency=self.stream_latency_sec,
            callback=self._input_callback,
        )
        self.bass_stream = sd.OutputStream(
            device=self.bass_device,
            samplerate=self.sample_rate,
            channels=2,
            dtype="float32",
            blocksize=self.blocksize,
            latency=self.stream_latency_sec,
            callback=self._output_callback(self.bass_queue),
        )
        self.vocal_stream = sd.OutputStream(
            device=self.vocal_device,
            samplerate=self.sample_rate,
            channels=2,
            dtype="float32",
            blocksize=self.blocksize,
            latency=self.stream_latency_sec,
            callback=self._output_callback(self.vocal_queue),
        )

        self.input_stream.start()
        self.bass_stream.start()
        self.vocal_stream.start()
        self._log("Live split started.")

    def stop(self):
        for stream in (self.input_stream, self.bass_stream, self.vocal_stream):
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass

        self.input_stream = None
        self.bass_stream = None
        self.vocal_stream = None

        while not self.bass_queue.empty():
            self.bass_queue.get_nowait()
        while not self.vocal_queue.empty():
            self.vocal_queue.get_nowait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Split Router (Bass + Vocal)")
        self.resize(1100, 700)

        self.playback = SplitPlaybackEngine(status_cb=self.set_status)
        self.live = LiveSplitEngine(status_cb=self.set_status)

        self.songs: list[str] = []
        self.current_index: int = -1
        self.user_seeking = False

        self._build_ui()
        self._apply_styles()
        self.refresh_devices()

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_player_state)
        self.poll_timer.start(300)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(14)

        title = QLabel("Audio Split Router")
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Signal-based real-time bass/vocal split (No AI)")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        source_group = QGroupBox("Song Folder")
        source_layout = QGridLayout(source_group)
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Choose folder containing songs...")

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_folder)

        load_btn = QPushButton("Load Songs")
        load_btn.clicked.connect(self.load_songs)

        source_layout.addWidget(self.folder_edit, 0, 0, 1, 4)
        source_layout.addWidget(browse_btn, 0, 4)
        source_layout.addWidget(load_btn, 0, 5)
        layout.addWidget(source_group)

        middle_layout = QHBoxLayout()

        playlist_group = QGroupBox("Playlist")
        playlist_layout = QVBoxLayout(playlist_group)
        self.playlist = QListWidget()
        self.playlist.itemDoubleClicked.connect(self.play_selected)
        playlist_layout.addWidget(self.playlist)

        transport = QHBoxLayout()
        prev_btn = QPushButton("Prev")
        prev_btn.clicked.connect(self.play_previous)
        play_btn = QPushButton("Play")
        play_btn.clicked.connect(self.play_selected)
        pause_btn = QPushButton("Pause")
        pause_btn.clicked.connect(self.pause_song)
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self.stop_song)
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(self.play_next)
        for btn in (prev_btn, play_btn, pause_btn, stop_btn, next_btn):
            transport.addWidget(btn)
        playlist_layout.addLayout(transport)

        progress_row = QHBoxLayout()
        self.elapsed_label = QLabel("00:00")
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self.seek_slider.sliderReleased.connect(self._on_seek_released)
        self.seek_slider.sliderMoved.connect(self._on_seek_moved)
        self.total_label = QLabel("00:00")
        progress_row.addWidget(self.elapsed_label)
        progress_row.addWidget(self.seek_slider, 1)
        progress_row.addWidget(self.total_label)
        playlist_layout.addLayout(progress_row)

        devices_group = QGroupBox("Audio Routing")
        devices_layout = QGridLayout(devices_group)

        self.bass_combo = QComboBox()
        self.vocal_combo = QComboBox()
        self.input_combo = QComboBox()
        self.buffer_combo = QComboBox()
        self.buffer_combo.addItems(["256", "512", "1024", "2048", "4096"])
        self.buffer_combo.setCurrentText("2048")
        self.buffer_combo.currentTextChanged.connect(self.apply_audio_tuning)

        self.latency_slider = QSlider(Qt.Orientation.Horizontal)
        self.latency_slider.setRange(10, 300)
        self.latency_slider.setValue(80)
        self.latency_slider.valueChanged.connect(self._update_latency_label)
        self.latency_slider.sliderReleased.connect(self.apply_audio_tuning)
        self.latency_value_label = QLabel("80 ms")

        self.bass_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.bass_gain_slider.setRange(0, 200)
        self.bass_gain_slider.setValue(100)
        self.bass_gain_slider.valueChanged.connect(self.apply_gain_controls)
        self.bass_gain_value_label = QLabel("1.00x")

        self.vocal_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.vocal_gain_slider.setRange(0, 200)
        self.vocal_gain_slider.setValue(100)
        self.vocal_gain_slider.valueChanged.connect(self.apply_gain_controls)
        self.vocal_gain_value_label = QLabel("1.00x")

        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.clicked.connect(self.refresh_devices)

        devices_layout.addWidget(QLabel("Bass Output Device (Bluetooth):"), 0, 0)
        devices_layout.addWidget(self.bass_combo, 0, 1)
        devices_layout.addWidget(QLabel("Vocal Output Device (Laptop):"), 1, 0)
        devices_layout.addWidget(self.vocal_combo, 1, 1)
        devices_layout.addWidget(QLabel("Live Input Device:"), 2, 0)
        devices_layout.addWidget(self.input_combo, 2, 1)
        devices_layout.addWidget(QLabel("Buffer (samples):"), 3, 0)
        devices_layout.addWidget(self.buffer_combo, 3, 1)

        latency_row = QHBoxLayout()
        latency_row.addWidget(self.latency_slider, 1)
        latency_row.addWidget(self.latency_value_label)
        devices_layout.addWidget(QLabel("Latency:"), 4, 0)
        devices_layout.addLayout(latency_row, 4, 1)

        bass_gain_row = QHBoxLayout()
        bass_gain_row.addWidget(self.bass_gain_slider, 1)
        bass_gain_row.addWidget(self.bass_gain_value_label)
        devices_layout.addWidget(QLabel("Bass Gain:"), 5, 0)
        devices_layout.addLayout(bass_gain_row, 5, 1)

        vocal_gain_row = QHBoxLayout()
        vocal_gain_row.addWidget(self.vocal_gain_slider, 1)
        vocal_gain_row.addWidget(self.vocal_gain_value_label)
        devices_layout.addWidget(QLabel("Vocal Gain:"), 6, 0)
        devices_layout.addLayout(vocal_gain_row, 6, 1)

        devices_layout.addWidget(refresh_btn, 7, 1)

        live_controls = QHBoxLayout()
        live_start_btn = QPushButton("Start Live Split")
        live_stop_btn = QPushButton("Stop Live Split")
        live_start_btn.clicked.connect(self.start_live)
        live_stop_btn.clicked.connect(self.stop_live)
        live_controls.addWidget(live_start_btn)
        live_controls.addWidget(live_stop_btn)
        devices_layout.addLayout(live_controls, 8, 0, 1, 2)

        middle_layout.addWidget(playlist_group, 3)
        middle_layout.addWidget(devices_group, 2)

        layout.addLayout(middle_layout)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("status")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self._update_latency_label(self.latency_slider.value())
        self.apply_audio_tuning()
        self.apply_gain_controls()

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                background-color: #12151e;
                color: #e9edf6;
                font-family: Segoe UI;
                font-size: 11pt;
            }
            QGroupBox {
                border: 1px solid #2d3550;
                border-radius: 10px;
                margin-top: 12px;
                padding: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #8ad8ff;
            }
            QPushButton {
                background-color: #2d61e5;
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3a74ff;
            }
            QPushButton:pressed {
                background-color: #1f4fc3;
            }
            QLineEdit, QListWidget, QComboBox {
                background-color: #1a1f2f;
                border: 1px solid #2f3753;
                border-radius: 8px;
                padding: 6px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2a3150;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #54b4ff;
                border: 1px solid #9fd8ff;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QLabel#subtitle {
                color: #8ea0c0;
                margin-bottom: 6px;
            }
            QLabel#status {
                background-color: #171d2d;
                border: 1px solid #2f3753;
                border-radius: 8px;
                padding: 10px;
                color: #b7f0ff;
            }
            """
        )

    def set_status(self, msg: str):
        self.status_label.setText(msg)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Song Folder")
        if folder:
            self.folder_edit.setText(folder)
            self.load_songs()

    def load_songs(self):
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Invalid Folder", "Please choose a valid folder.")
            return

        songs = []
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                songs.append(path)

        self.songs = songs
        self.playlist.clear()

        for song in songs:
            item = QListWidgetItem(os.path.basename(song))
            self.playlist.addItem(item)

        if songs:
            self.current_index = 0
            self.playlist.setCurrentRow(0)
            self.set_status(f"Loaded {len(songs)} songs.")
        else:
            self.current_index = -1
            self.set_status("No supported audio files found.")

    def _selected_device_index(self, combo: QComboBox):
        data = combo.currentData()
        if data is None:
            raise RuntimeError("Device not selected")
        return int(data)

    def refresh_devices(self):
        self.bass_combo.clear()
        self.vocal_combo.clear()
        self.input_combo.clear()

        devices = sd.query_devices()
        default_in, default_out = sd.default.device

        best_bass_row = -1
        best_vocal_row = -1
        best_input_row = -1

        for idx, dev in enumerate(devices):
            name = dev["name"]
            out_channels = int(dev["max_output_channels"])
            in_channels = int(dev["max_input_channels"])

            if out_channels > 0:
                text = f"[{idx}] {name}"
                self.bass_combo.addItem(text, idx)
                self.vocal_combo.addItem(text, idx)

                row = self.bass_combo.count() - 1
                if "bluetooth" in name.lower() and best_bass_row == -1:
                    best_bass_row = row

                if idx == default_out and best_vocal_row == -1:
                    best_vocal_row = row

            if in_channels > 0:
                text = f"[{idx}] {name}"
                self.input_combo.addItem(text, idx)
                row = self.input_combo.count() - 1
                if idx == default_in and best_input_row == -1:
                    best_input_row = row

        if self.bass_combo.count() > 0:
            self.bass_combo.setCurrentIndex(best_bass_row if best_bass_row != -1 else 0)
        if self.vocal_combo.count() > 0:
            self.vocal_combo.setCurrentIndex(best_vocal_row if best_vocal_row != -1 else 0)
        if self.input_combo.count() > 0:
            self.input_combo.setCurrentIndex(best_input_row if best_input_row != -1 else 0)

        self.set_status("Audio devices refreshed.")

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds_int = max(0, int(seconds))
        minutes = seconds_int // 60
        rem = seconds_int % 60
        return f"{minutes:02d}:{rem:02d}"

    def _update_latency_label(self, value: int):
        self.latency_value_label.setText(f"{int(value)} ms")

    def apply_audio_tuning(self):
        try:
            blocksize = int(self.buffer_combo.currentText())
        except ValueError:
            blocksize = 2048
        latency_ms = int(self.latency_slider.value())
        self.playback.set_tuning(blocksize=blocksize, latency_ms=latency_ms)
        self.live.set_tuning(blocksize=max(128, blocksize // 2), latency_ms=latency_ms)

    def apply_gain_controls(self):
        bass_gain = float(self.bass_gain_slider.value()) / 100.0
        vocal_gain = float(self.vocal_gain_slider.value()) / 100.0
        self.bass_gain_value_label.setText(f"{bass_gain:.2f}x")
        self.vocal_gain_value_label.setText(f"{vocal_gain:.2f}x")
        self.playback.set_gains(bass_gain=bass_gain, vocal_gain=vocal_gain)
        self.live.set_gains(bass_gain=bass_gain, vocal_gain=vocal_gain)

    def _on_seek_pressed(self):
        self.user_seeking = True

    def _on_seek_moved(self, value: int):
        self.elapsed_label.setText(self._format_time(value))

    def _on_seek_released(self):
        self.user_seeking = False
        target_sec = float(self.seek_slider.value())
        self.playback.seek_seconds(target_sec)
        duration = self.playback.get_duration_seconds()
        self.total_label.setText(self._format_time(duration))

    def _prepare_playback(self, song_path: str):
        bass_device = self._selected_device_index(self.bass_combo)
        vocal_device = self._selected_device_index(self.vocal_combo)

        self.playback.set_devices(bass_device, vocal_device)
        self.playback.load_file(song_path)

    def play_selected(self):
        if not self.songs:
            self.set_status("No songs loaded.")
            return

        row = self.playlist.currentRow()
        if row < 0:
            row = self.current_index if self.current_index >= 0 else 0
            self.playlist.setCurrentRow(row)

        self.current_index = row
        song = self.songs[row]

        try:
            self._prepare_playback(song)
            self.playback.play()
            duration = self.playback.get_duration_seconds()
            self.seek_slider.setRange(0, max(0, int(duration)))
            self.seek_slider.setValue(0)
            self.elapsed_label.setText("00:00")
            self.total_label.setText(self._format_time(duration))
            self.set_status(f"Playing: {os.path.basename(song)}")
        except Exception as exc:
            QMessageBox.critical(self, "Playback Error", str(exc))
            self.set_status("Playback failed.")

    def pause_song(self):
        try:
            self.playback.pause()
        except Exception as exc:
            self.set_status(f"Pause failed: {exc}")

    def stop_song(self):
        try:
            self.playback.stop()
            self.set_status("Stopped.")
        except Exception as exc:
            self.set_status(f"Stop failed: {exc}")

    def play_next(self):
        if not self.songs:
            return
        self.current_index = (self.current_index + 1) % len(self.songs)
        self.playlist.setCurrentRow(self.current_index)
        self.play_selected()

    def play_previous(self):
        if not self.songs:
            return
        self.current_index = (self.current_index - 1) % len(self.songs)
        self.playlist.setCurrentRow(self.current_index)
        self.play_selected()

    def start_live(self):
        try:
            input_dev = self._selected_device_index(self.input_combo)
            bass_dev = self._selected_device_index(self.bass_combo)
            vocal_dev = self._selected_device_index(self.vocal_combo)

            input_info = sd.query_devices(input_dev)
            sr = int(input_info.get("default_samplerate", 48000))

            self.live.configure(
                input_device=input_dev,
                bass_device=bass_dev,
                vocal_device=vocal_dev,
                sample_rate=sr,
            )
            self.live.start()
            self.set_status(f"Live split active @ {sr} Hz")
        except Exception as exc:
            QMessageBox.critical(self, "Live Split Error", str(exc))
            self.set_status("Could not start live split.")

    def stop_live(self):
        try:
            self.live.stop()
            self.set_status("Live split stopped.")
        except Exception as exc:
            self.set_status(f"Live stop failed: {exc}")

    def _poll_player_state(self):
        if not self.user_seeking:
            pos = self.playback.get_position_seconds()
            duration = self.playback.get_duration_seconds()
            self.seek_slider.setRange(0, max(0, int(duration)))
            self.seek_slider.setValue(max(0, int(pos)))
            self.elapsed_label.setText(self._format_time(pos))
            self.total_label.setText(self._format_time(duration))

        if self.playback.file_done_event.is_set():
            self.playback.file_done_event.clear()
            self.play_next()

    def closeEvent(self, event):
        self.live.stop()
        self.playback.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Split Router")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
