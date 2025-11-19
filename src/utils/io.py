import json
import os
import numpy as np
from pathlib import Path

def ensure_dir(path):
	if path and not os.path.exists(path):
		os.makedirs(path, exist_ok=True)


def read_text(path):
	with open(path, "r", encoding="utf-8") as f:
		return f.read()

def load_jsonl(path):
	out = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			if line.strip():
				out.append(json.loads(line))
	return out

def _make_json_safe(obj):
	if isinstance(obj, Path):
		return str(obj)
	if isinstance(obj, dict):
		return {k: _make_json_safe(v) for k, v in obj.items()}
	if isinstance(obj, list):
		return [_make_json_safe(x) for x in obj]
	if isinstance(obj, tuple):
		return [_make_json_safe(x) for x in obj]
	if isinstance(obj, (np.integer,)):
		return int(obj)
	if isinstance(obj, (np.floating,)):
		return float(obj)
	if isinstance(obj, np.ndarray):
		return obj.tolist()

	return obj

def save_jsonl(objects, path):
	with open(path, "w", encoding="utf-8") as f:
		for obj in objects:
			safe = _make_json_safe(obj)
			f.write(json.dumps(safe, ensure_ascii=False) + "\n")

def save_embeddings(path, array):
	ensure_dir(Path(path).parent)
	np.save(path, array)

def load_embeddings(path):
	return np.load(path)