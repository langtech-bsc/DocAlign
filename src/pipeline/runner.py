"""Main pipeline orchestration."""
from pathlib import Path
import os
from multiprocessing import Pool

from .document_loader import DocumentPair
from .aligner import DocumentAligner
from .output_builder import OutputBuilder


def process_subdir(subdir, model, model_path, aligner, output_builder):
    try:
        print(f"\n{'=' * 60}\nProcessing {subdir}\n{'=' * 60}")

        # Load document pair
        doc_pair = DocumentPair.from_directory(subdir)
        doc_pair.load_embeddings(model, model_path)

        # Align
        results = aligner.align(doc_pair)

        # Save outputs
        output_builder.save_alignment(doc_pair, results, subdir)
    except Exception as e:
        print(f"❌ Error processing {subdir}: {e}")


def run_pipeline(input_dir, output_dir, model="labse", model_path=None, dtw="approx"):
    """Process all document pairs in input directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all subdirectories
    subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not subdirs:
        subdirs = [input_dir]

    # Initialize components
    aligner = DocumentAligner(model, model_path, dtw)
    output_builder = OutputBuilder(output_dir)

    # Prepare args for each task
    tasks = [
        (subdir, model, model_path, aligner, output_builder)
        for subdir in subdirs
    ]
    # compute number of workers (cpus), it shouldn't be more than the number of subdirectories to process, shouldn't
    # be more than the number of cpus (minus one)
    n_workers = min(len(subdirs), max(1, os.cpu_count() - 1))

    # compute all the tasks
    with Pool(processes=n_workers) as pool:
        pool.starmap(process_subdir, tasks)

    print(f"\n{'=' * 60}\nPipeline complete!\n{'=' * 60}")
