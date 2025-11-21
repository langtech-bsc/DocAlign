"""Main pipeline orchestration."""
from pathlib import Path
from .document_loader import DocumentPair
from .aligner import DocumentAligner
from .output_builder import OutputBuilder


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
    
    # Process each directory
    for subdir in subdirs:
        try:
            print(f"\n{'='*60}\nProcessing {subdir.name}\n{'='*60}")
            
            # Load document pair
            doc_pair = DocumentPair.from_directory(subdir)
            doc_pair.load_embeddings(model, model_path)
            
            # Align
            results = aligner.align(doc_pair)
            
            # Save outputs
            output_builder.save_alignment(doc_pair, results, subdir.name)
            
        except Exception as e:
            print(f"❌ Error processing {subdir.name}: {e}")
            continue
    
    print(f"\n{'='*60}\nPipeline complete!\n{'='*60}")