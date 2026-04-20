# Backend Audio Processing Principle

This document explains only what happens to audio internally while the app runs.

## 1) Processing model: streaming blocks

Audio is processed as continuous fixed-size blocks (for example 1024/2048 samples), not as one full waveform at once.

For each block, the runtime pipeline is:

$$
	ext{input block} \rightarrow \text{stereo decomposition} \rightarrow \text{DSP split} \rightarrow \text{gain/volume} \rightarrow \text{dual output queues} \rightarrow \text{two device callbacks}
$$

This supports real-time behavior and low memory usage.

---

## 2) Stereo decomposition (mid/side domain)

For left/right channels $L[n], R[n]$:

$$
M[n] = 0.5\,(L[n]+R[n]), \quad S[n] = 0.5\,(L[n]-R[n])
$$

- $M[n]$ (mid) represents center content.
- $S[n]$ (side) represents stereo difference content.

This is used because vocals are commonly center-weighted in many mixes.

---

## 3) Bass branch DSP

Bass is extracted by low-pass filtering the mid signal:

$$
B[n] = LPF_{180\text{ Hz}}(M[n])
$$

Implementation uses a 4th-order Butterworth IIR filter in second-order-section form.

Effect: keeps low-frequency energy and rejects most mids/highs.

---

## 4) Vocal branch DSP

First, a center-emphasized signal is formed:

$$
C[n] = M[n] - 0.25\,S[n]
$$

Then vocal-range band-pass filtering is applied:

$$
V[n] = BPF_{220\text{ Hz} \rightarrow 5000\text{ Hz}}(C[n])
$$

Effect: emphasizes speech/singing band and suppresses deep bass + very high content.

---

## 5) Stateful IIR continuity across blocks

Both filters keep internal state vectors (`zi`) between blocks.

Without state carry-over, every block would reset filter history and introduce discontinuities/clicks. With preserved state, filtering is continuous as if processing one long stream.

---

## 6) Per-path scaling and protection

After filtering, each branch has two multiplicative stages:

1. **Gain** (band emphasis): bass/vocal gain
2. **Volume** (final loudness): bass/vocal volume

Then hard limiting by clipping:

$$
y[n] = \mathrm{clip}(y[n], -1, 1)
$$

This ensures safe float audio range for output streams.

---

## 7) Producer-consumer audio runtime

The app uses decoupled execution:

- **Producer** thread/callback computes DSP blocks.
- **Consumer** output callbacks send blocks to each device.

Communication uses two queues:

- bass queue
- vocal queue

If a callback arrives before data is ready, a zero block is emitted (underflow safety). This avoids callback crash/stall.

---

## 8) Dual-device output path

Each output device has its own callback and clock domain. The app maps each dequeued block to the device channel count (mono/stereo adaptation), then writes to that endpoint.

So backend routing is:

$$
\{B[n],V[n]\} \rightarrow \{Q_b,Q_v\} \rightarrow \{D_b,D_v\}
$$

where $Q_b,Q_v$ are queues and $D_b,D_v$ are selected output devices.

---

## 9) Start synchronization (pre-roll)

Before output streams start, the engine waits until both queues contain initial blocks (pre-roll).

Purpose: reduce startup race where one device starts on silence while the other starts with valid data.

---

## 10) File mode backend path

File mode signal chain:

$$
	ext{file reader} \rightarrow \text{DSP per block} \rightarrow \text{queues} \rightarrow \text{2 output callbacks}
$$

The reader advances playback position; at end-of-file it signals completion for track transition.

---

## 11) Live mode backend path

Live mode signal chain:

$$
	ext{input callback} \rightarrow \text{DSP per block} \rightarrow \text{queues} \rightarrow \text{2 output callbacks}
$$

Difference from file mode: the source is incoming device audio instead of file blocks. DSP core remains the same.

---

## 12) Why separation is approximate

This is non-AI spectral/stereo-domain processing, so it is deterministic and fast, but not stem-perfect.

- Bass branch: frequency-isolated low content.
- Vocal branch: center-emphasized vocal-band content.

Quality depends on original mix characteristics (vocal panning, mastering, arrangement overlap).
