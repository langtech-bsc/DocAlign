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
from src.postprocess import rebuild

from sentence_transformers import SentenceTransformer

def process_folder(subdir, output_dir, model, model_path, dtw):
    subdir = Path(subdir)
    output_dir = Path(output_dir)

    jsonl_files = list(subdir.glob("*.jsonl"))
    if len(jsonl_files) != 2:
        print(f"⚠ Skipping {subdir}: expected 2 jsonl files, found {len(jsonl_files)}")
        return

    print(f"[PID {os.getpid()}] Processing {subdir.name}")
    # Reuse your existing inner code (everything inside the for-loop)
    # Just extract it into this function cleanly
    # You would move all the code from the for-loop into here.


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
		output_dir,
		model="labse",
		model_path=None,
		dtw="approx"
	):
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)

	print("Loading data...")
	for subdir in sorted([d for d in input_dir.iterdir() if d.is_dir()]):
		jsonl_files = list(subdir.glob("*.jsonl"))

		if len(jsonl_files) != 2:
			print(f"⚠ Skipping {subdir}: expected exactly 2 jsonl files, found {len(jsonl_files)}")
			continue

		outdir = output_dir / f"{subdir.name}"
	
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

		print(f"Saving paths and sentence_alignments to {outdir}")
		outfile1 = outdir / f"{subdir.name}_meta.jsonl"
		outfile1.parent.mkdir(parents=True, exist_ok=True)
		safe_out =io._make_json_safe(out)
		io.save_jsonl([safe_out], outfile1)

		sent_index = {
			(entry["paragraph_A"], entry["paragraph_B"]): entry["sentence_pairs"]
			for entry in all_sentence_alignments_sorted
		}

		# Build paragraph-level aligned blocks
		paragraph_blocks = rebuild.build_paragraph_blocks(
			all_paragraph_alignments_sorted=[
				{"src_idx": i, "tgt_idx": j}
				for (i, j) in para_path
			],
			sentence_alignment_index=sent_index
		)

		chunks = rebuild.build_chunks(paragraph_blocks, max_words=2500)


		# Save chunks alongside main output
		outfile2 = outdir / f"{subdir.name}_chunks.jsonl"
		outfile2.parent.mkdir(parents=True, exist_ok=True)
		print(f"Saving chunked output to {outfile2}")
		io.save_jsonl(chunks, outfile2)



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