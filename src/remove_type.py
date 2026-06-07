import json
import sys

INPUT_FILE = "../outputs/checkpoint.json"
TYPE_TO_REMOVE = sys.argv[1]

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

before = len(data)

data = [x for x in data if x.get("type") != TYPE_TO_REMOVE]

with open(INPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Removed {before - len(data)} entries")