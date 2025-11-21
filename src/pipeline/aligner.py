"""Core alignment logic."""
import tqdm
from src.utils import collapse_path
from src.align import coarse_align, fine_align_debug


class DocumentAligner:
    """Handles paragraph and sentence-level alignment."""
    
    def __init__(self, model, model_path=None, dtw_mode="approx"):
        self.model = model
        self.model_path = model_path
        self.dtw_mode = dtw_mode
    
    def align(self, doc_pair):
        """
        Align a DocumentPair and return alignment results.
        
        Returns:
            dict with keys: para_path, collapsed, sentence_alignments
        """
        print("Running paragraph-level DTW...")
        para_path = coarse_align.align_paragraphs(
            doc_pair.emb_a, doc_pair.emb_b, mode=self.dtw_mode
        )
        
        collapsed = collapse_path.collapse_path_to_ranges(para_path)
        print("Collapsed paragraph ranges:", collapsed)
        
        print("Computing sentence-level alignments...")
        sentence_alignments = self._align_sentences(
            doc_pair.paragraphs_a,
            doc_pair.paragraphs_b,
            collapsed
        )
        
        return {
            "para_path": para_path,
            "collapsed": collapsed,
            "sentence_alignments": sentence_alignments
        }
    
    def _align_sentences(self, paragraphs_a, paragraphs_b, collapsed):
        """Align sentences within collapsed paragraph ranges."""
        alignments = []
        
        for source_indices, target_indices in tqdm.tqdm(collapsed):
            combined_a = " ".join(paragraphs_a[i] for i in source_indices)
            combined_b = " ".join(paragraphs_b[i] for i in target_indices)
            
            sentence_pairs = fine_align_debug.align_sentence_level(
                combined_a, combined_b,
                model=self.model,
                model_path=self.model_path
            )
            
            if sentence_pairs:
                alignments.append({
                    "paragraph_A": tuple(source_indices),
                    "paragraph_B": tuple(target_indices),
                    "sentence_pairs": sentence_pairs,
                })
        
        return sorted(alignments, key=lambda x: (x["paragraph_A"], x["paragraph_B"]))