"""Build and save alignment outputs."""
from pathlib import Path
from src.utils import io
from src.postprocess import rebuild


class OutputBuilder:
    """Handles building and saving alignment outputs."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_alignment(self, doc_pair, alignment_results, name):
        """Save alignment metadata and chunks."""
        subdir = self.output_dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        self._save_metadata(doc_pair, alignment_results, subdir, name)
        
        # Build and save chunks
        chunks = self._build_chunks(alignment_results)
        self._save_chunks(chunks, subdir, name)
        
        print(f"✓ Saved {len(chunks)} chunks to {subdir}")
    
    def _save_metadata(self, doc_pair, results, subdir, name):
        """Save alignment metadata to JSONL."""
        meta = {
            "input_A": str(doc_pair.path_a),
            "input_B": str(doc_pair.path_b),
            "model": "labse",  # Could be passed in
            "dtw": "approx",   # Could be passed in
            "alignment_path": results["para_path"],
            "sentence_alignments": results["sentence_alignments"]
        }
        
        outfile = subdir / f"{name}_meta.jsonl"
        io.save_jsonl([io._make_json_safe(meta)], outfile)
    
    def _build_chunks(self, results, max_words=2500):
        """Build aligned chunks from sentence alignments."""
        sent_index = {
            (entry["paragraph_A"], entry["paragraph_B"]): entry["sentence_pairs"]
            for entry in results["sentence_alignments"]
        }
        
        paragraph_blocks = rebuild.build_paragraph_blocks(
            all_paragraph_alignments_sorted=[
                {"src_idx": tuple(src), "tgt_idx": tuple(tgt)}
                for src, tgt in results["collapsed"]
            ],
            sentence_alignment_index=sent_index
        )
        
        return rebuild.build_chunks(paragraph_blocks, max_words=max_words)
    
    def _save_chunks(self, chunks, subdir, name):
        """Save chunks to JSONL."""
        outfile = subdir / f"{name}_chunks.jsonl"
        io.save_jsonl(chunks, outfile)