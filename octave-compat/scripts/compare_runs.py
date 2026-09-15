"""Compare the MATLAB(master)-under-Octave run against the Python-port run."""
import glob, os, sys
import numpy as np
import nibabel as nib

M = "/home/user/roast/example"
P = "/home/user/roast-py/example"


def find(d, suffix):
    hits = sorted(glob.glob(os.path.join(d, "*" + suffix)))
    return hits[0] if hits else None


def load(path):
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float64), img


def stats(name, a):
    print(f"  {name:<28} shape={a.shape} min={np.nanmin(a):.6g} "
          f"max={np.nanmax(a):.6g} mean={np.nanmean(a):.6g}")


def compare_volume(label, pa, pb):
    print(f"\n### {label}")
    if pa is None or pb is None:
        print(f"  MISSING  master={pa}  python={pb}")
        return
    print(f"  master: {os.path.basename(pa)}")
    print(f"  python: {os.path.basename(pb)}")
    a, ia = load(pa)
    b, ib = load(pb)
    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH {a.shape} vs {b.shape}")
        return
    stats("master", a)
    stats("python", b)
    if not np.allclose(ia.affine, ib.affine, atol=1e-6):
        print("  affine differs:")
        print("   master\n", ia.affine, "\n   python\n", ib.affine)
    else:
        print("  affine: identical")
    d = a - b
    denom = max(np.nanmax(np.abs(a)), 1e-30)
    print(f"  max |diff| = {np.nanmax(np.abs(d)):.6g}   "
          f"({100*np.nanmax(np.abs(d))/denom:.4f}% of master max)")
    print(f"  RMS diff   = {np.sqrt(np.nanmean(d**2)):.6g}")
    fa, fb = a.ravel(), b.ravel()
    ok = np.isfinite(fa) & np.isfinite(fb)
    if ok.sum() > 1 and fa[ok].std() > 0 and fb[ok].std() > 0:
        print(f"  Pearson r  = {np.corrcoef(fa[ok], fb[ok])[0,1]:.8f}")
    nz = np.abs(a) > 0.01 * denom
    if nz.sum():
        rel = np.abs(d[nz]) / np.abs(a[nz])
        print(f"  over |master|>1% of max ({nz.sum()} vox): "
              f"median rel.err {np.median(rel)*100:.4f}%, "
              f"p95 {np.percentile(rel,95)*100:.4f}%")


def brain_stats(label, emag_path, mask_path):
    if emag_path is None or mask_path is None:
        return None
    e, _ = load(emag_path)
    m, _ = load(mask_path)
    if m.ndim == 4:
        gm = m[..., 0] > 0
        wm = m[..., 1] > 0
        brain = gm | wm
    else:
        brain = (m == 1) | (m == 2)
    if brain.sum() == 0 or e.shape != brain.shape:
        return None
    v = e[brain]
    v = v[np.isfinite(v)]
    print(f"  {label:<8} brain voxels={brain.sum()}  "
          f"mean|E|={v.mean():.6f}  median={np.median(v):.6f}  "
          f"p99={np.percentile(v,99):.6f}  max={v.max():.6f} V/m")
    return v


print("=" * 72)
print("ROAST: master (MATLAB/Octave) vs claude/matlab-python-translation (Python)")
print("example: MNI152_T1_1mm.nii, recipe Fp1 +1 mA / P4 -1 mA (defaults)")
print("=" * 72)

for label, suffix in [("Segmentation masks", "_masks.nii"),
                      ("Electric potential (V)", "_v.nii"),
                      ("E-field magnitude (V/m)", "_emag.nii"),
                      ("E-field vector (V/m)", "_e.nii")]:
    compare_volume(label, find(M, suffix), find(P, suffix))

print("\n### |E| inside grey+white matter")
em, ep = find(M, "_emag.nii"), find(P, "_emag.nii")
mm, mp = find(M, "_masks.nii"), find(P, "_masks.nii")
va = brain_stats("master", em, mm)
vb = brain_stats("python", ep, mp)
if va is not None and vb is not None:
    print(f"  difference in mean |E|: {abs(va.mean()-vb.mean()):.3e} V/m "
          f"({100*abs(va.mean()-vb.mean())/max(va.mean(),1e-30):.4f}%)")

print("\n### files produced")
for tag, d in (("master", M), ("python", P)):
    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(d, "*"))
                   if not f.endswith((".zip",)))
    print(f"  {tag}: {len(names)} files")
    for n in names:
        print(f"    {n}")
