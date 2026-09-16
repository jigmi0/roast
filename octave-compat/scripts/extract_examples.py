"""Extract each example command from runExamples.m, keyed by its cmd number.

runExamples.m wraps every example in try/catch whose handler prints
'error at running cmd N:', which is what identifies the example.  Multi-line
commands (continuation dots) are joined into one line.

    python3 octave-compat/scripts/extract_examples.py > master_cmds.json
"""
import json
import re
import sys

src = open("runExamples.m").read()
blocks = re.findall(r"try\s*\n(.*?)\ncatch ME.*?error at running cmd (\d+):", src, re.S)
cmds = {}
for body, n in blocks:
    lines = [l.strip() for l in body.strip().splitlines()]
    cmd = " ".join(l.rstrip(".") if l.endswith("...") else l for l in lines)
    cmds[int(n)] = re.sub(r"\s+", " ", cmd).strip()
json.dump(cmds, sys.stdout, indent=0, sort_keys=True)
