"""Check which microphone the bot is using and whether it hears you clearly.

Run from the project folder (virtual environment active):
    python scripts/mic_test.py          # list devices + test the default mic
    python scripts/mic_test.py 3        # test input device number 3

It records 8 seconds. Stay SILENT for the first 3 seconds (measures background
noise), then talk normally. Nothing is saved or sent anywhere.
"""

import math
import struct
import sys
import time

import pyaudio

RATE, CHUNK, SECONDS = 16000, 1600, 8  # 0.1 s chunks

pa = pyaudio.PyAudio()

print("INPUT DEVICES (microphones)")
default_in = pa.get_default_input_device_info()
default_out = pa.get_default_output_device_info()
for i in range(pa.get_device_count()):
    d = pa.get_device_info_by_index(i)
    if d.get("maxInputChannels", 0) > 0:
        mark = "  <- default" if i == default_in["index"] else ""
        print(f"  {i:>2}: {d['name']}{mark}")
print("\nOUTPUT DEVICES (speakers / headphones)")
for i in range(pa.get_device_count()):
    d = pa.get_device_info_by_index(i)
    if d.get("maxOutputChannels", 0) > 0:
        mark = "  <- default" if i == default_out["index"] else ""
        print(f"  {i:>2}: {d['name']}{mark}")

index = int(sys.argv[1]) if len(sys.argv) > 1 else default_in["index"]
name = pa.get_device_info_by_index(index)["name"]
print(f"\nTesting microphone {index}: {name}")
print("Stay SILENT for 3 seconds, then talk normally...\n")

stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                 input_device_index=index, frames_per_buffer=CHUNK)
quiet, loud = [], []
start = time.time()
for n in range(int(RATE / CHUNK * SECONDS)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    samples = struct.unpack(f"{len(data) // 2}h", data)
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    db = 20 * math.log10(max(rms, 1) / 32768)  # 0 dB = max, about -60 dB = silence
    (quiet if n < 30 else loud).append(db)
    phase = "silent" if n < 30 else "TALK  "
    bar = "#" * max(0, int((db + 60) / 1.5))
    print(f"  {time.time() - start:4.1f}s {phase} {db:6.1f} dB |{bar}")
stream.stop_stream()
stream.close()
pa.terminate()

noise = sorted(quiet)[len(quiet) // 2]
speech = sorted(loud)[int(len(loud) * 0.8)]
gap = speech - noise
print(f"\nBackground noise: {noise:.1f} dB | Your speech: {speech:.1f} dB | Gap: {gap:.1f} dB")
if speech < -45:
    print("PROBLEM: your voice is very quiet. Wrong microphone, muted, or mic blocked by")
    print("Windows/macOS privacy settings. Try another device number from the list.")
elif gap < 12:
    print("PROBLEM: background noise is close to your voice (fan, traffic, TV, echo).")
    print("Use a headset mic, move away from the fan, or raise vad_min_volume to 0.7.")
else:
    print("GOOD: the bot should hear you clearly with this microphone.")
if index != default_in["index"]:
    print(f"\nTo use this mic, set in config/agent.yaml:\naudio:\n  input_device_index: {index}")
