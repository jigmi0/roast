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
# Run the two implementations at the same time? Off by default; see the
# note in main() about getDP being OOM-killed when two solves overlap.
PARALLEL = os.environ.get("ROAST_SUITE_PARALLEL") == "1"
# Runs correlating below this are left on disk instead of deleted.
KEEP_BELOW_R = float(os.environ.get("ROAST_SUITE_KEEP_BELOW_R", "0.95"))
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


def find_masks(d, shape):
    """The tissue masks matching this run's geometry.

    A subject has one _masks.nii per derived image (plain, _padded<N>,
    resampled), so match on shape - the padded variants differ in size - and
    prefer the most recently written among equals.
    """
    for f in sorted(glob.glob(os.path.join(d, "*_masks.nii")),
                    key=os.path.getmtime, reverse=True):
        try:
            if nib.load(f).shape == shape:
                return f
        except Exception:
            continue
    return None


def _stats(a, b, sel, prefix, row):
    x, y = a[sel], b[sel]
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return
    rel = np.abs(x - y) / np.maximum(np.abs(x), 1e-12)
    row[prefix + "_n"] = int(sel.sum())
    row[prefix + "_r"] = float(np.corrcoef(x, y)[0, 1])
    row[prefix + "_mean_m"] = float(x.mean())
    row[prefix + "_mean_p"] = float(y.mean())
    row[prefix + "_mean_pct"] = float(100 * abs(x.mean() - y.mean()) / max(abs(x.mean()), 1e-30))
    row[prefix + "_med_rel_pct"] = float(100 * np.median(rel))
    row[prefix + "_p95_rel_pct"] = float(100 * np.percentile(rel, 95))
    row[prefix + "_frac_over5pct"] = float((rel > 0.05).mean())


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
    row["status"] = "ok"
    ok = np.isfinite(a) & np.isfinite(b)
    _stats(a, b, ok, "emag", row)

    # the number that actually matters: |E| inside grey + white matter
    mk = find_masks(MEX, a.shape)
    if mk is not None:
        m = load(mk)
        brain = ((m == 1) | (m == 2)) & ok
        if brain.sum() > 1:
            row["masks"] = os.path.basename(mk)
            _stats(a, b, brain, "brain", row)

    if mv and pv:
        x, y = load(mv), load(pv)
        o = np.isfinite(x) & np.isfinite(y)
        if o.sum() > 1:
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
        # The two sides are independent and write to different directories, so
        # running them concurrently is tempting - but getDP solves the system
        # with a MUMPS LU factorisation that peaks near 8 GB on the padded
        # models, and two at once gets one of them OOM-killed, which then looks
        # exactly like a port defect. Sequential is the default for that reason;
        # set ROAST_SUITE_PARALLEL=1 only on a box with RAM to spare.
        e = dict(env, ROAST_CMD=CMDS[str(n)])
        master_cmd = ["octave-cli", "--no-gui",
                      os.path.join(M, "octave-compat", "scripts", "run_one_master.m")]
        port_cmd = [PY, "python/examples/run_examples.py", str(n)]
        fm = open(f"{SP}/ex{n}_master.log", "w")
        fp = open(f"{SP}/ex{n}_python.log", "w")
        pm = subprocess.Popen(master_cmd, cwd=M, env=e, stdout=fm, stderr=subprocess.STDOUT)
        if PARALLEL:
            pp = subprocess.Popen(port_cmd, cwd=P, env=env, stdout=fp, stderr=subprocess.STDOUT)
        for proc, key in ((pm, "master"),):
            try:
                rec[key + "_rc"] = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                rec[key] = "timeout"
        if not PARALLEL:
            pp = subprocess.Popen(port_cmd, cwd=P, env=env, stdout=fp, stderr=subprocess.STDOUT)
        try:
            rec["python_rc"] = pp.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pp.kill()
            rec["python"] = "timeout"
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
        # Keep the outputs of a run that disagrees, so the anomaly can actually
        # be investigated afterwards; only clean up the ones that agree.
        rr = rec.get("brain_r", rec.get("emag_r"))
        if rec.get("status") == "ok" and rr is not None and rr < KEEP_BELOW_R:
            print("   !! r=%.4f < %s; keeping outputs for inspection" % (rr, KEEP_BELOW_R),
                  flush=True)
        else:
            for t in new_m | new_p:
                cleanup(t)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
