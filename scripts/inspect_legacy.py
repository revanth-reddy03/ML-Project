import json
from pathlib import Path

legacy_dir = Path("notebooks/legacy_backup")
for f in sorted(legacy_dir.glob("*.ipynb")):
    with open(f, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    cells = data.get("cells", [])
    print(f"\n==================== {f.name} ====================")
    print(f"Total cells: {len(cells)}")
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))[:120].replace("\n", " ")
        outputs = len(c.get("outputs", [])) if c["cell_type"] == "code" else "N/A"
        print(f"[{i}] {c['cell_type']} (outputs={outputs}): {src}")
