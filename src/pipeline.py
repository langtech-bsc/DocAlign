#!/usr/bin/env python3

import os
import json
import yaml
import tqdm
import argparse
import numpy as np
import numpy as np
from pathlib import Path
from fastdtw import fastdtw
import dtaidistance
from dtaidistance import dtw as dtw_exact
from typing import List, Tuple, Dict
from src.utils import io, sentence_split
from src.encode import embed_paragraphs, embed_sentences
from src.align import coarse_align, fine_align

from sentence_transformers import SentenceTransformer



# --- Utility functions -----------------------------------------------------

# def load_jsonl_paragraphs(path: str) -> Tuple[List[str], List[Dict]]:
#     """
#     Load a JSONL file where each line is a JSON object with at least:
#       - "doc_id", "para_id", "lang", "text"
#     Returns:
#       - paragraphs: list of text strings, ordered by increasing para_id
#       - meta: list of dicts (same length) with original metadata (doc_id, para_id, lang)
#     """
#     records = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             obj = json.loads(line)
#             # Ensure minimal fields present
#             if "para_id" not in obj:
#                 raise ValueError(f"Record missing para_id: {obj}")
#             if "text" not in obj:
#                 raise ValueError(f"Record missing text: {obj}")
#             records.append(obj)

#     # Sort by para_id to ensure deterministic ordering (int)
#     try:
#         records.sort(key=lambda r: int(r["para_id"]))
#     except Exception:
#         # fallback to current order if para_id not int-like
#         pass

#     paragraphs = [r["text"] for r in records]
#     meta = [{"doc_id": r.get("doc_id"), "para_id": r.get("para_id"), "lang": r.get("lang")} for r in records]
#     return paragraphs, meta

# # --- DTW wrappers ----------------------------------------------------------

# def align_with_dtw(emb_a: np.ndarray, emb_b: np.ndarray) -> List[Tuple[int, int]]:
#     """
#     Approximate DTW using fastdtw. Returns list of (i_a, i_b).
#     """
#     if fastdtw is None:
#         raise ImportError("fastdtw not installed. Install with `pip install fastdtw` or use --dtw exact if dtaidistance is available.")
#     def cos_dist(x, y):
#         # x, y are 1-D numpy arrays
#         return 1.0 - float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
#     distance, path = fastdtw(emb_a, emb_b, dist=cos_dist)
#     # fastdtw returns list of (idx_in_a, idx_in_b)
#     return path


# def align_with_exact_dtw(emb_a: np.ndarray, emb_b: np.ndarray) -> List[Tuple[int, int]]:
#     """
#     Exact DTW using dtaidistance (O(N*M)). Returns list of (i_a, i_b).
#     """
#     if dtw_exact is None:
#         raise ImportError("dtaidistance not installed. Install with `pip install dtaidistance` to use exact DTW.")
#     # normalize to cosine distance matrix: 1 - cosine_sim
#     # dtaidistance expects a 2D numpy array of distances
#     # Compute pairwise distance matrix safely (may be large)
#     na = emb_a.shape[0]
#     nb = emb_b.shape[0]
#     # compute norms
#     norms_a = np.linalg.norm(emb_a, axis=1) + 1e-9
#     norms_b = np.linalg.norm(emb_b, axis=1) + 1e-9
#     # dot product matrix
#     dot = np.dot(emb_a, emb_b.T)  # shape (na, nb)
#     dist_matrix = 1.0 - (dot / (norms_a[:, None] * norms_b[None, :]))
#     # dtaidistance provides warping path from distance matrix
#     path = dtw_exact.warping_path_from_distance_matrix(dist_matrix)
#     # path is a list of (i_a, i_b)
#     return path


# # --- I/O helpers -----------------------------------------------------------

# def save_path_json(path: List[Tuple[int, int]], out_path: str):
#     """Save DTW path as JSON list of objects {"a": i, "b": j}"""
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump([{"a": int(i), "b": int(j)} for i, j in path], f, ensure_ascii=False, indent=2)


# def save_embeddings(emb: np.ndarray, out_path: str):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     np.save(out_path, emb)


# --- CLI / Orchestration --------------------------------------------------


def run(par_file_a: str, par_file_b: str, out_dir: str,
		model_name: str = None, model_obj = None,
		dtw_mode: str = "approx", save_emb: bool = True):
	"""
	Main pipeline runner.
	- par_file_a, par_file_b: input JSONL paragraph files
	- out_dir: directory to save outputs (path.json, embeddings)
	- model_name or model_obj: either a model name to load or an already created model
	- dtw_mode: "approx" (fastdtw) or "exact" (dtaidistance)
	"""
	os.makedirs(out_dir, exist_ok=True)

	print("Loading paragraphs...")
	paras_a, meta_a = load_jsonl_paragraphs(par_file_a)
	paras_b, meta_b = load_jsonl_paragraphs(par_file_b)
	print(f"Loaded {len(paras_a)} paragraphs from A and {len(paras_b)} paragraphs from B.")

	# Build/load embedding model
	if model_obj is None:
		if model_name is None:
			model_name = "sentence-transformers/all-MiniLM-L6-v2"
		print(f"Loading embedding model: {model_name}")
		model = build_model(model_name)
	else:
		model = model_obj

	# Compute embeddings
	print("Computing embeddings for A...")
	emb_a = get_embeddings(paras_a, model)
	print("Computing embeddings for B...")
	emb_b = get_embeddings(paras_b, model)

	if save_emb:
		emb_a_path = os.path.join(out_dir, "emb_a.npy")
		emb_b_path = os.path.join(out_dir, "emb_b.npy")
		print(f"Saving embeddings to {emb_a_path} and {emb_b_path}")
		save_embeddings(emb_a, emb_a_path)
		save_embeddings(emb_b, emb_b_path)

	# Run DTW alignment
	print(f"Running DTW ({dtw_mode}) ...")
	if dtw_mode in ("approx", "fast", "fastdtw"):
		path = align_with_dtw(emb_a, emb_b)
	elif dtw_mode in ("exact", "dtaidistance"):
		path = align_with_exact_dtw(emb_a, emb_b)
	else:
		raise ValueError("dtw_mode must be 'approx' or 'exact'")

	print(f"DTW path length: {len(path)}")

	# Save path
	out_path = os.path.join(out_dir, "dtw_path.json")
	save_path_json(path, out_path)
	print(f"Saved DTW path to: {out_path}")

	# Also save small metadata for reference
	meta_out = {
		"left_file": par_file_a,
		"right_file": par_file_b,
		"n_paras_a": len(paras_a),
		"n_paras_b": len(paras_b),
		"dtw_path_len": len(path)
	}
	with open(os.path.join(out_dir, "meta_run.json"), "w", encoding="utf-8") as f:
		json.dump(meta_out, f, ensure_ascii=False, indent=2)

	return path, (emb_a, emb_b), (paras_a, paras_b), (meta_a, meta_b)

def run_pipeline(
		input_dir,
		output_path,
		model="labse",
		model_path=None,
		dtw="approx"
	):
	"""
	Run the document-level alignment pipeline from config specs.
	Steps:
	  1. Load JSONL paragraph data
	  2. Compute paragraph embeddings
	  3. Paragraph-level DTW alignment
	  4. Save alignment results
	"""
	print("Loading data...")
	root = Path(input_dir)
	jsonl_files = list(root.glob("*.jsonl"))
	path_A = next((f for f in jsonl_files if "A" in f.name), None)
	path_B = next((f for f in jsonl_files if "B" in f.name), None)

	doc_A = io.load_jsonl(path_A)
	doc_B = io.load_jsonl(path_B)

	paragraphs_A = [entry["text"] for entry in doc_A]
	paragraphs_B = [entry["text"] for entry in doc_B]

	emb_A_path = root / "A.para.npy"
	emb_B_path = root / "B.para.npy"

	print("Embedding paragraphs...")
	# A
	if emb_A_path.exists():
		print(f"Loading cached embeddings for A from {emb_A_path}")
		emb_A = io.load_embeddings(emb_A_path)
		print("Shape of A:", emb_A.shape)
	else:
		emb_A = embed_paragraphs.get_paragraph_embeddings(
			paragraphs_A, model=model, model_path=model_path
		)
		io.save_embeddings(emb_A_path, emb_A)

	# B
	if emb_B_path.exists():
		print(f"Loading cached embeddings for B from {emb_B_path}")
		emb_B = io.load_embeddings(emb_B_path)
		print("Shape of B:", emb_B.shape)
	else:
		emb_B = embed_paragraphs.get_paragraph_embeddings(
			paragraphs_B, model=model, model_path=model_path
		)
		io.save_embeddings(emb_B_path, emb_B)

	print("Running paragraph-level DTW...")
	para_path = coarse_align.align_paragraphs(emb_A, emb_B, mode=dtw)

	print("Working out sentence-level alignments")
	all_sentence_alignments = []
	for (i, j) in tqdm.tqdm(para_path):
		paraA = paragraphs_A[i]

		paraB = paragraphs_B[j]
		sentence_pairs = fine_align.align_sentence_level(
			paraA,
			paraB,
			model=model,
			model_path=model_path,
		)

		all_sentence_alignments.append({
			"paragraph_A": i,
			"paragraph_B": j,
			"sentence_pairs": sentence_pairs,
		})

		all_sentence_alignments_sorted = sorted(
			all_sentence_alignments,
			key=lambda x: (x["paragraph_A"], x["paragraph_B"])
		)
	# # Prepare output
	io.ensure_dir(os.path.dirname(output_path))

	out = {
		"input_A": path_A,
		"input_B": path_B,
		"model": model,
		"dtw": dtw,
		"alignment_path": para_path,
		"sentence_alignments": all_sentence_alignments_sorted
	}

	print(f"Saving output to {output_path}")
	io.save_jsonl([out], output_path)

	return out


def run_pipeline_from_config(config):
	io.ensure_dir(os.path.dirname(config["output"]))
	return run_pipeline(
		input_dir=config.get("input"),
		output_path=config["output"],
		model=config.get("model", "labse"),
		model_path=config.get("model_path"),
		dtw=config.get("dtw")
	)


def main():
	parser = argparse.ArgumentParser(description="Run the data cleaning pipeline.")
	parser.add_argument("--config", type=str, required=True,
						help="Path to YAML config file")
	args = parser.parse_args()

	with open(args.config, "r", encoding="utf-8") as f:
		config = yaml.safe_load(f)

	run_pipeline_from_config(config)


if __name__ == "__main__":
	main()