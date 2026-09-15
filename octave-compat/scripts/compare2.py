import glob, os
import numpy as np, nibabel as nib, scipy.io as sio

M, P = "/home/user/roast/example", "/home/user/roast-py/example"
def f(d, suf):
    h = sorted(glob.glob(os.path.join(d, "*"+suf))); return h[0] if h else None
def L(p): return np.asarray(nib.load(p).dataobj, dtype=np.float64)

TISSUE = {1:"grey", 2:"white", 3:"CSF", 4:"bone", 5:"skin", 6:"air"}
ma, pa = L(f(M,"_masks.nii")), L(f(P,"_masks.nii"))
print("=== SEGMENTATION MASKS (after touch-up) ===")
same = (ma == pa)
print(f"  identical voxels: {same.sum():,} / {ma.size:,}  ({100*same.mean():.4f}%)")
print(f"  {'tissue':<8}{'master vox':>12}{'python vox':>12}{'diff':>10}{'diff %':>9}   Dice")
for k, nm in TISSUE.items():
    A, B = ma == k, pa == k
    dice = 2*(A & B).sum() / max(A.sum()+B.sum(), 1)
    d = int(B.sum()) - int(A.sum())
    print(f"  {nm:<8}{A.sum():>12,}{B.sum():>12,}{d:>10,}{100*d/max(A.sum(),1):>8.3f}%   {dice:.6f}")

print("\n=== ELECTRODE / GEL MASKS ===")
for suf in ("_mask_elec.nii", "_mask_gel.nii"):
    a, b = L(f(M,suf)), L(f(P,suf))
    A, B = a > 0, b > 0
    dice = 2*(A & B).sum()/max(A.sum()+B.sum(),1)
    print(f"  {suf:<16} master {A.sum():>8,} vox   python {B.sum():>8,} vox   Dice {dice:.6f}")

print("\n=== MESH ===")
for nm, d in (("master", M), ("python", P)):
    mf = [x for x in glob.glob(os.path.join(d,"*.mat")) if "roastResult" not in x
          and "seg8" not in x and "rmask" not in x and "usedElecArea" not in x
          and "roastOptions" not in x]
    if mf:
        z = sio.loadmat(mf[0])
        print(f"  {nm}: nodes {z['node'].shape[0]:,}  elements {z['elem'].shape[0]:,}  faces {z['face'].shape[0]:,}")
for nm, d in (("master", M), ("python", P)):
    q = f(d, "_usedElecArea.mat")
    if q: print(f"  {nm} electrode contact areas (mm^2): {np.asarray(sio.loadmat(q)['area_elecNeeded']).ravel()}")

print("\n=== FIELDS, restricted to voxels finite in BOTH ===")
masks = ma
brain = (masks == 1) | (masks == 2)
head = masks > 0
for label, suf in (("voltage V", "_v.nii"), ("|E| V/m", "_emag.nii")):
    a, b = L(f(M,suf)), L(f(P,suf))
    ok = np.isfinite(a) & np.isfinite(b)
    for rname, reg in (("whole head", head & ok), ("grey+white", brain & ok)):
        x, y = a[reg], b[reg]
        r = np.corrcoef(x, y)[0,1]
        rel = np.abs(x-y)/np.maximum(np.abs(x), 1e-12)
        print(f"  {label:<10} {rname:<11} n={reg.sum():>9,}  r={r:.6f}  "
              f"mean {x.mean():.6f} vs {y.mean():.6f}  "
              f"median rel.diff {100*np.median(rel):.3f}%  p95 {100*np.percentile(rel,95):.3f}%")
