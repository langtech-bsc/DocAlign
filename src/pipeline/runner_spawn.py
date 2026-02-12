"""Main pipeline orchestration."""
from pathlib import Path
import os
import multiprocessing as mp
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
	mp.set_start_method('spawn', force=True)
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
	tasks = [
		(subdir, model, model_path, aligner, output_builder)
		for subdir in to_process
	]
	# compute number of workers (cpus), it shouldn't be more than the number of subdirectories to process, shouldn't
	# be more than the number of cpus (minus one)
	n_workers = min(len(subdirs), max(1, os.cpu_count() - 1))

	# compute all the tasks
	with mp.Pool(processes=n_workers) as pool:
		pool.starmap(process_subdir, tasks)

	print(f"\n{'=' * 60}\nPipeline complete!\n{'=' * 60}", flush=True)
