"""Execute a notebook, report errors and total time, save executed copy."""
import sys, time, nbformat
from nbclient import NotebookClient

src, dst = sys.argv[1], sys.argv[2]
nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name="python3", allow_errors=True, record_timing=True)
t0 = time.time()
client.execute()
elapsed = time.time() - t0
nbformat.write(nb, dst)
n_err = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != "code": continue
    for o in c.get("outputs", []):
        if o.output_type == "error":
            n_err += 1
            print(f"--- ERROR in cell {i}: {o.ename}: {o.evalue}")
            print("\n".join(o.traceback[-8:])[:3000])
            print("--- source head:\n" + "\n".join(c.source.splitlines()[:6]))
print(f"executed {src} in {elapsed:.0f}s, {n_err} errors, saved {dst}")
