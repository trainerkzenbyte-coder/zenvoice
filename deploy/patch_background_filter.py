"""Adds the background speech filter (app/noise_guard.py) to app/phone_bot.py.

Run once from the project folder (on the server: ~/voice-agent):

    python3 deploy/patch_background_filter.py

It checks your phone_bot.py first and changes nothing if it doesn't look as expected.
A backup is kept as app/phone_bot.py.bak. Running it twice is harmless.
"""

import re
import shutil
import sys
from pathlib import Path

path = Path("app/phone_bot.py")
if not path.exists():
    sys.exit("app/phone_bot.py not found: run this from the project folder (e.g. cd ~/voice-agent)")
src = path.read_text(encoding="utf-8")
if "make_background_filter" in src:
    sys.exit("Already added: nothing to do.")

# 1) create the meter + filter right after the voice gate (or right after the transport)
anchor = re.search(r"^( *)voice_gate = noise_guard\.make_voice_gate\(label\)[^\n]*\n", src, re.M)
if not anchor:
    sys.exit("Could not find the voice gate line in app/phone_bot.py: nothing changed. Send me the file.")
indent = anchor.group(1)
src = src[:anchor.end()] + (
    f"{indent}bg_meter, bg_filter = noise_guard.make_background_filter(label)  "
    "# crowd / traffic / lecture voices never interrupt\n") + src[anchor.end():]

# 2) meter right before speech-to-text, filter right after the transcript fixer
pipe = re.search(r"^( *)stt,[^\n]*\n\1fixer,[^\n]*\n", src, re.M)
if not pipe:
    sys.exit("Could not find 'stt,' followed by 'fixer,' in the pipeline: nothing changed. Send me the file.")
ind = pipe.group(1)
lines = pipe.group(0).splitlines(keepends=True)
new = (f"{ind}*([bg_meter] if bg_meter else []),   # measures how loud the candidate is\n"
       + lines[0] + lines[1]
       + f"{ind}*([bg_filter] if bg_filter else []),  # drops words from other people (crowd, traffic, TV)\n")
src = src[:pipe.start()] + new + src[pipe.end():]

compile(src, str(path), "exec")  # never write a broken file
shutil.copyfile(path, path.with_suffix(".py.bak"))
path.write_text(src, encoding="utf-8")
print("Done: background speech filter added to app/phone_bot.py (backup: app/phone_bot.py.bak)")
