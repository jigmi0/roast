import time, traceback
from roast import roast
t0 = time.time()
try:
    roast(show=False)
    print("\n=== ROAST RETURNED OK ===")
except Exception:
    print("\n=== ROAST ERROR ===")
    traceback.print_exc()
print("=== ELAPSED %.1f s ===" % (time.time() - t0))
