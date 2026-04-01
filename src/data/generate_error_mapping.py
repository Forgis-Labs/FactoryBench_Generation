"""Regenerate root_cause_to_error_mapping.json from root_causes.json + protocol CSVs."""
import csv
import json
import re
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]

# Load root causes
rc_path = repo_root / "data/labelling/rca/root_causes.json"
root_causes = json.load(open(rc_path, encoding="utf-8"))

# Load runtime errors CSV into a lookup: error_code -> row
def load_protocols(csv_path):
    lookup = {}
    for row in csv.DictReader(open(csv_path, encoding="utf-8")):
        # Handle BOM in first field name
        code_field = next(k for k in row if "Error Code" in k or "error_code" in k.lower())
        raw_code = row[code_field].strip()
        # Normalise: remove all spaces -> "C4A1" style key
        norm = re.sub(r'\s+', '', raw_code)
        lookup[norm] = row
    return lookup

runtime = load_protocols(repo_root / "data/protocols/UR3_Runtime_Errors.csv")

def get_error_row(code):
    norm = re.sub(r'\s+', '', code)
    return runtime.get(norm)

mapping = []
for rc in root_causes:
    primary = rc.get("primary_error_code")
    if not primary:
        continue
    row = get_error_row(primary)
    if row is None:
        print(f"WARNING: {primary} not found in CSV (fault_id={rc['fault_id']})")
        continue

    # Field names vary slightly due to BOM
    desc_field = next((k for k in row if k.lower().strip('\ufeff') == 'description'), None)
    expl_field = next((k for k in row if k.lower().strip('\ufeff') == 'explanation'), None)
    sugg_field = next((k for k in row if k.lower().strip('\ufeff') == 'suggestion'), None)

    mapping.append({
        "fault_id": rc["fault_id"],
        "root_cause": rc["root_cause"],
        "root_cause_description": rc["description"],
        "primary_error_code": primary,
        "error_description": row.get(desc_field, "").strip() if desc_field else "",
        "error_explanation": row.get(expl_field, "").strip() if expl_field else "",
        "error_suggestion": row.get(sugg_field, "").strip() if sugg_field else "",
    })

out_path = repo_root / "data/knowledge_graph/root_cause_to_error_mapping.json"
json.dump(mapping, open(out_path, "w", encoding="utf-8"), indent=2)
print(f"Written {len(mapping)} entries to {out_path.name}")
