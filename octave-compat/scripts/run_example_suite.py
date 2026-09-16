"""Run ROAST examples on both implementations, compare, and clean up.

For each example number given, runs master (under Octave) and the Python
port concurrently, compares the resulting E-field volumes, appends a row to
results.jsonl, then deletes that run's outputs - a single example generates
~1.3 GB across the two trees.

Paths come from ROAST_MASTER_DIR / ROAST_PORT_DIR / ROAST_SUITE_WORKDIR;
example commands from ROAST_CMDS_JSON (see extract_examples.py).

    python3 run_example_suite.py 5 7 9 11
"""
import json, os, re, subprocess, sys, time, glob, shutil
import numpy as np, nibabel as nib

# Working directory for logs, results and the interpolation cache.
SP = os.environ.get("ROAST_SUITE_WORKDIR", os.path.abspath("roast-suite-work"))
os.makedirs(SP, exist_ok=True)
# The two checkouts to compare: master and the Python port.
M = os.environ.get("ROAST_MASTER_DIR", "/home/user/roast")
P = os.environ.get("ROAST_PORT_DIR", "/home/user/roast-py")
MEX, PEX = os.path.join(M, "example"), os.path.join(P, "example")
PY = "/home/user/venv-roast/bin/python"
CMDS = json.load(open(os.environ.get("ROAST_CMDS_JSON", "master_cmds.json")))
TAG_RE = re.compile(r"_(\d{8}T\d{6})")

env = dict(os.environ, ROAST_PYTHON=PY, ROAST_TSI_WORKDIR=os.path.join(SP, "tsi_work"),
           QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg", ROAST_MATLAB_CMD="octave-cli")


def tags(d):
    return {m.group(1) for f in os.listdir(d) if (m := TAG_RE.search(f))}


def run_master(n, timeout):
    e = dict(env, ROAST_CMD=CMDS[str(n)])
    with open(f"{SP}/ex{n}_master.log", "w") as fh:
        return subprocess.run(["octave-cli", "--no-gui", os.path.join(M, "octave-compat", "scripts", "run_one_master.m")],
                              cwd=M, env=e, stdout=fh, stderr=subprocess.STDOUT,
                              timeout=timeout).returncode


def run_python(n, timeout):
    with open(f"{SP}/ex{n}_python.log", "w") as fh:
        return subprocess.run([PY, "python/examples/run_examples.py", str(n)],
                              cwd=P, env=env, stdout=fh, stderr=subprocess.STDOUT,
                              timeout=timeout).returncode


def load(p):
    return np.asarray(nib.load(p).dataobj, dtype=np.float64)


def newest(d, tag, suffix):
    hits = [f for f in glob.glob(os.path.join(d, f"*{tag}*{suffix}"))]
    return hits[0] if hits else None


def compare(n, mt, pt):
    row = {"example": n, "desc": CMDS[str(n)][:70]}
    me, pe = newest(MEX, mt, "_emag.nii"), newest(PEX, pt, "_emag.nii")
    mv, pv = newest(MEX, mt, "_v.nii"), newest(PEX, pt, "_v.nii")
    if not (me and pe):
        row["status"] = "no _emag output"
        return row
    a, b = load(me), load(pe)
    if a.shape != b.shape:
        row["status"] = f"shape {a.shape} vs {b.shape}"
        return row
    ok = np.isfinite(a) & np.isfinite(b)
    row["status"] = "ok"
    row["n_finite"] = int(ok.sum())
    row["emag_r"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    row["emag_mean_m"] = float(a[ok].mean())
    row["emag_mean_p"] = float(b[ok].mean())
    row["emag_mean_pct"] = 100 * abs(a[ok].mean() - b[ok].mean()) / max(a[ok].mean(), 1e-30)
    rel = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(a[ok]), 1e-12)
    row["emag_med_rel_pct"] = float(100 * np.median(rel))
    if mv and pv:
        x, y = load(mv), load(pv)
        o = np.isfinite(x) & np.isfinite(y)
        row["v_r"] = float(np.corrcoef(x[o], y[o])[0, 1])
    return row


def cleanup(tag):
    for d in (MEX, PEX):
        for f in glob.glob(os.path.join(d, f"*{tag}*")):
            try:
                os.remove(f)
            except OSError:
                pass


def main(nums, timeout=3600):
    out = os.path.join(SP, "results.jsonl")
    for n in nums:
        t0 = time.time()
        before_m, before_p = tags(MEX), tags(PEX)
        print(f"[{time.strftime('%H:%M:%S')}] example {n}: {CMDS[str(n)][:70]}", flush=True)
        rec = {"example": n}
        # the two sides are independent and write to different directories,
        # so run them concurrently instead of back to back
        e = dict(env, ROAST_CMD=CMDS[str(n)])
        fm = open(f"{SP}/ex{n}_master.log", "w")
        fp = open(f"{SP}/ex{n}_python.log", "w")
        pm = subprocess.Popen(["octave-cli", "--no-gui", os.path.join(M, "octave-compat", "scripts", "run_one_master.m")],
                              cwd=M, env=e, stdout=fm, stderr=subprocess.STDOUT)
        pp = subprocess.Popen([PY, "python/examples/run_examples.py", str(n)],
                              cwd=P, env=env, stdout=fp, stderr=subprocess.STDOUT)
        for proc, key in ((pm, "master"), (pp, "python")):
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                rec[key] = "timeout"
        fm.close(); fp.close()
        new_m, new_p = tags(MEX) - before_m, tags(PEX) - before_p
        if new_m and new_p:
            rec.update(compare(n, sorted(new_m)[-1], sorted(new_p)[-1]))
        else:
            rec["status"] = f"no new output (master {len(new_m)}, python {len(new_p)})"
        rec["secs"] = round(time.time() - t0, 1)
        print("   ->", json.dumps({k: v for k, v in rec.items() if k != "desc"}), flush=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        for t in new_m | new_p:
            cleanup(t)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
