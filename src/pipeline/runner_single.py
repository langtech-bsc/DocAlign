"""Main pipeline orchestration."""
from pathlib import Path
import os
from multiprocessing import Pool

from .document_loader import DocumentPair
from .aligner import DocumentAligner
from .output_builder import OutputBuilder

def already_processed(subdir, output_dir):
	# Example: consider the subdir processed if alignment.json exists
	expected_file = output_dir / subdir.name / f"{subdir.name}_chunks.jsonl"
	return expected_file.exists()


def process_subdir(subdir, model, model_path, aligner, output_builder):
	try:
		print(f"\n{'=' * 60}\nProcessing {subdir}\n{'=' * 60}", flush=True)

		# Load document pair
		doc_pair = DocumentPair.from_directory(subdir)
		doc_pair.load_embeddings(model, model_path)

		# Align
		results = aligner.align(doc_pair)

		# Save outputs
		output_builder.save_alignment(doc_pair, results, subdir)
	except Exception as e:
		print(f"❌ Error processing {subdir}: {e}", flush=True)


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

	# Filter out subdirs that are already processed
	to_process = [
		d for d in subdirs 
		if not already_processed(d, output_dir)
	]

	if not to_process:
		print("Everything already processed.")
		return

	print(f"Skipping {len(subdirs) - len(to_process)} previously processed subdirectories.")
	# Prepare args for each task
		# Sequential execution – no multiprocessing
	for subdir in to_process:
		process_subdir(subdir, model, model_path, aligner, output_builder)

	print(f"\n{'=' * 60}\nPipeline complete!\n{'=' * 60}", flush=True)