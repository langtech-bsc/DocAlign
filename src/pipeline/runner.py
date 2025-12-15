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

def process_subdir_wrapper(args):
    """Wrapper to unpack arguments and handle initialization per process."""
    subdir, model, model_path, dtw, output_dir = args
    
    try:
        print(f"\n{'=' * 60}\nProcessing {subdir}\n{'=' * 60}", flush=True)
        
        # Initialize components inside the worker process (important for 'spawn')
        aligner = DocumentAligner(model, model_path, dtw)
        output_builder = OutputBuilder(output_dir)
        
        # Load document pair
        doc_pair = DocumentPair.from_directory(subdir)
        doc_pair.load_embeddings(model, model_path)
        
        # Align
        results = aligner.align(doc_pair)
        
        # Save outputs
        output_builder.save_alignment(doc_pair, results, subdir)
        
        print(f"✓ Successfully processed {subdir.name}", flush=True)
        return (subdir.name, "success", None)
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ Error processing {subdir.name}: {error_msg}", flush=True)
        return (subdir.name, "failed", error_msg)

def run_pipeline(input_dir, output_dir, model="labse", model_path=None, dtw="approx", use_multiprocessing=True):
    """Process all document pairs in input directory."""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all subdirectories
    subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not subdirs:
        subdirs = [input_dir]
    
    print(f"Found {len(subdirs)} total subdirectories")
    
    # Filter out subdirs that are already processed
    to_process = [
        d for d in subdirs 
        if not already_processed(d, output_dir)
    ]
    
    if not to_process:
        print("Everything already processed.")
        return
    
    print(f"Skipping {len(subdirs) - len(to_process)} previously processed subdirectories.")
    print(f"Will process {len(to_process)} subdirectories.")
    
    if use_multiprocessing:
        # MULTIPROCESSING MODE (for CPU or multiple GPUs)
        mp.set_start_method('spawn', force=True)
        
        # Prepare args for each task
        tasks = [
            (subdir, model, model_path, dtw, output_dir)
            for subdir in to_process
        ]
        
        # Limit workers for GPU - each process loads model into GPU memory
        max_gpu_workers = 8  # Adjust based on your GPU memory and model size
        n_workers = min(max_gpu_workers, max(1, os.cpu_count() - 1))
        print(f"Using {n_workers} workers (limited for GPU memory management)\n")
        
        # compute all the tasks and collect results
        results = []
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(process_subdir_wrapper, tasks)
    else:
        # SEQUENTIAL MODE (recommended for single GPU)
        print("Processing sequentially (recommended for GPU)\n")
        
        # Initialize components once
        aligner = DocumentAligner(model, model_path, dtw)
        output_builder = OutputBuilder(output_dir)
        
        results = []
        for i, subdir in enumerate(to_process, 1):
            print(f"\n[{i}/{len(to_process)}] Processing {subdir.name}", flush=True)
            
            try:
                # Load document pair
                doc_pair = DocumentPair.from_directory(subdir)
                doc_pair.load_embeddings(model, model_path)
                
                # Align
                align_results = aligner.align(doc_pair)
                
                # Save outputs
                output_builder.save_alignment(doc_pair, align_results, subdir)
                
                print(f"✓ Successfully processed {subdir.name}", flush=True)
                results.append((subdir.name, "success", None))
                
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                print(f"❌ Error processing {subdir.name}: {error_msg}", flush=True)
                results.append((subdir.name, "failed", error_msg))
    
    # Analyze results
    successful = [r for r in results if r[1] == "success"]
    failed = [r for r in results if r[1] == "failed"]
    
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete!")
    print(f"Successfully processed: {len(successful)}/{len(to_process)}")
    if failed:
        print(f"Failed: {len(failed)}")
        print("\nFailed subdirectories:")
        for name, status, error in failed:
            print(f"  - {name}")
    print(f"{'=' * 60}\n", flush=True)
    
    return results