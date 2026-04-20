# Audio Split Router (No AI, Signal Processing Only)

This desktop app splits audio into:
- **Bass part** (low-frequency content)
- **Vocal part** (center-emphasized vocal-band content)

Then it routes:
- Bass -> selected Bluetooth/output device
- Vocal -> selected laptop/output device

It also supports **real-time live splitting** from an input device (mic/line-in) using DSP operations.

## Tech
- Python
- PySide6 (UI)
- NumPy + SciPy (signal processing)
- sounddevice + soundfile (audio I/O)

## Features
- Load songs from a folder
- Auto-play songs one-by-one
- Prev / Next / Pause / Stop
- Per-song progress bar + seek slider
- Select separate output devices for bass and vocal
- Buffer and latency controls
- Per-band gain controls (Bass Gain, Vocal Gain)
- Real-time live split mode
- Modern dark UI
- One-click run via BAT file

## Run (one click)
Double-click:
- **run_app.bat**

It will:
1. Create `.venv` if missing
2. Install dependencies
3. Launch the app

## Build EXE
Double-click:
- **build_exe.bat**

Then use:
- `dist\\AudioSplitRouter.exe`

## Notes
- Separation is DSP-based (filters + stereo operations), not source-separation AI.
- Best quality with stereo songs and proper device sample-rate support.
- If Bluetooth has delay, use low-latency devices/drivers when possible.
