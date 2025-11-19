#!/usr/bin/env python3
import os
import json
import yaml
import tqdm
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool

# Your imports
from src.utils import io, sentence_split
from src.encode import embed_paragraphs, embed_sentences
from src.align import coarse_align, fine_align
from src.postprocess import rebuild

# -----------------------------
# GLOBAL MODEL FOR EACH PROCESS
# -----------------------------
MODEL_GLOBAL = None
MODEL_NAME_GLOBAL = None
MODEL_PATH_GLOBAL = None

def init_worker(model_name, model_path):
    """
    Initializer for multiprocessing workers.
    Loads the model once per worker.
    """
    global MODEL_GLOBAL, MODEL_NAME_GLOBAL, MODEL_PATH_GLOBAL

    MODEL_NAME_GLOBAL = model_name
    MODEL_PATH_GLOBAL = model_path

    print(f"[Worker] Loading model: {model_name}")
    MODEL_GLOBAL = embed_paragraphs.load_model(model_name, model_path)
    print(f"[Worker] Model loaded.")


# -----------------------------
# PROCESS ONE SUBDIRECTORY
# -----------------------------
def process_single_subdir(args):
    """
    Runs the alignment pipeline on a single subdirectory.
    Designed to be multiprocessing-safe.
    """
    subdir, output_dir, model_name, model_path, dtw = args

    print(f"Processing {subdir}")

    jsonl_files = list(subdir.glob("*.jsonl"))
    if len(jsonl_files) != 2:
        return f"Skipped {subdir} (did not contain exactly 2 JSONL files)"

    # Detect A/B files
    path_A = next((f for f in jsonl_files if "A" in f.name), None)
    path_B = next((f for f in jsonl_files if "B" in f.name), None)
    if not path_A or not path_B:
        return f"Skipped {subdir}: could not identify A/B"

    outdir = output_dir / f"{subdir.name}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Load texts
    doc_A = io.load_jsonl(path_A)
    doc_B = io.load_jsonl(path_B)
    paragraphs_A = [x["text"] for x in doc_A]
    paragraphs_B = [x["text"] for x in doc_B]

    # Cached embedding paths
    emb_A_path = subdir / "A.para.npy"
    emb_B_path = subdir / "B.para.npy"

    # -----------------------------
    # PARAGRAPH EMBEDDINGS
    # -----------------------------
    if emb_A_path.exists():
        emb_A = io.load_embeddings(emb_A_path)
    else:
        emb_A = embed_paragraphs.get_paragraph_embeddings(
            paragraphs_A,
            model=MODEL_GLOBAL,
            model_path=model_path
        )
        io.save_embeddings(emb_A_path, emb_A)

    if emb_B_path.exists():
        emb_B = io.load_embeddings(emb_B_path)
    else:
        emb_B = embed_paragraphs.get_paragraph_embeddings(
            paragraphs_B,
            model=MODEL_GLOBAL,
            model_path=model_path
        )
        io.save_embeddings(emb_B_path, emb_B)

    # -----------------------------
    # PARAGRAPH-LEVEL ALIGNMENT
    # -----------------------------
    para_path = coarse_align.align_paragraphs(emb_A, emb_B, mode=dtw)

    # -----------------------------
    # SENTENCE-LEVEL ALIGNMENTS
    # -----------------------------
    all_sentence_alignments = []
    for (i, j) in para_path:
        paraA = paragraphs_A[i]
        paraB = paragraphs_B[j]

        sentence_pairs = fine_align.align_sentence_level(
            paraA,
            paraB,
            model=model_name,
            model_path=model_path
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

    # -----------------------------
    # SAVE META
    # -----------------------------
    out_meta_file = outdir / f"{subdir.name}_meta.jsonl"
    safe_out = io._make_json_safe({
        "input_A": str(path_A),
        "input_B": str(path_B),
        "model": model_name,
        "dtw": dtw,
        "alignment_path": para_path,
        "sentence_alignments": all_sentence_alignments_sorted
    })
    io.save_jsonl([safe_out], out_meta_file)

    # -----------------------------
    # PARAGRAPH-BLOCK REBUILDING
    # -----------------------------
    sent_index = {
        (entry["paragraph_A"], entry["paragraph_B"]): entry["sentence_pairs"]
        for entry in all_sentence_alignments_sorted
    }

    paragraph_blocks = rebuild.build_paragraph_blocks(
        all_paragraph_alignments_sorted=[
            {"src_idx": i, "tgt_idx": j} for (i, j) in para_path
        ],
        sentence_alignment_index=sent_index
    )

    chunks = rebuild.build_chunks(paragraph_blocks, max_words=2500)

    out_chunks_file = outdir / f"{subdir.name}_chunks.jsonl"
    io.save_jsonl(chunks, out_chunks_file)

    return f"Completed {subdir.name}"


# -----------------------------
# TOP-LEVEL PIPELINE
# -----------------------------
def run_pipeline(input_dir, output_dir, model="labse", model_path=None, dtw="approx"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subdirs)} subdirectories")

    # Create worker pool
    with Pool(
        processes=os.cpu_count() - 1,
        initializer=init_worker,
        initargs=(model, model_path)
    ) as pool:

        # Build argument packages
        args_list = [
            (subdir, output_dir, model, model_path, dtw)
            for subdir in subdirs
        ]

        # Run in parallel with progress bar
        results = list(tqdm.tqdm(pool.imap_unordered(process_single_subdir, args_list),
                                 total=len(args_list)))

    # Print worker results
    for r in results:
        print(r)


# -----------------------------
# CONFIG WRAPPER
# -----------------------------
def run_pipeline_from_config(config):
    return run_pipeline(
        input_dir=config["input"],
        output_dir=config["output"],
        model=config.get("model", "labse"),
        model_path=config.get("model_path"),
        dtw=config.get("dtw", "approx")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_pipeline_from_config(config)


if __name__ == "__main__":
    main()
