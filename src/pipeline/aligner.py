"""Core alignment logic."""
import tqdm
from src.utils import collapse_path
from src.align import coarse_align, fine_align_spans


class DocumentAligner:
    """Handles paragraph and sentence-level alignment."""

    def __init__(self, model, model_path=None, dtw_mode="approx", thresholds=None):
        self.model = model
        self.model_path = model_path
        self.dtw_mode = dtw_mode
        self.thresholds = thresholds or {}

    def align(self, doc_pair):
        """
        Align a DocumentPair and return alignment results.

        Returns:
            dict with keys: para_path, collapsed, sentence_alignments, dtw_cost
        """
        print("Running paragraph-level DTW...", flush=True)
        para_path, dtw_cost, dist_matrix = coarse_align.align_paragraphs(
            doc_pair.emb_a, doc_pair.emb_b, mode=self.dtw_mode
        )
        print(f"DTW normalized cost: {dtw_cost:.4f}", flush=True)

        collapsed = collapse_path.collapse_path_to_ranges(para_path)
        print(f"Collapsed into {len(collapsed)} paragraph group(s)", flush=True)

        # Filter paragraph groups by cosine similarity
        para_threshold = self.thresholds.get("para_sim_threshold", 0.0)
        if para_threshold > 0:
            collapsed = self._filter_paragraph_groups(
                collapsed, doc_pair.emb_a, doc_pair.emb_b,
                dist_matrix, para_threshold
            )

        print("Computing sentence-level alignments...", flush=True)
        sentence_alignments = self._align_sentences(
            doc_pair.paragraphs_a,
            doc_pair.paragraphs_b,
            collapsed
        )

        return {
            "para_path": para_path,
            "collapsed": collapsed,
            "sentence_alignments": sentence_alignments,
            "dtw_cost": dtw_cost,
        }

    def _filter_paragraph_groups(self, collapsed, emb_a, emb_b,
                                  dist_matrix, threshold):
        """Drop collapsed paragraph groups whose mean cosine similarity
        falls below *threshold*.

        For each (src_indices, tgt_indices) group, compute the average
        cosine similarity across all (i, j) pairs.  If dist_matrix is
        available (exact DTW mode), look up directly; otherwise compute
        per-pair on the fly (approx mode).
        """
        filtered = []
        for src_indices, tgt_indices in collapsed:
            sims = []
            for i in src_indices:
                for j in tgt_indices:
                    if dist_matrix is not None:
                        sims.append(1.0 - float(dist_matrix[i, j]))
                    else:
                        sims.append(1.0 - coarse_align.cosine_distance(
                            emb_a[i], emb_b[j]))

            mean_sim = sum(sims) / len(sims) if sims else 0.0

            if mean_sim >= threshold:
                filtered.append((src_indices, tgt_indices))
            else:
                print(f"  Dropped paragraph group src={src_indices} "
                      f"tgt={tgt_indices} (sim={mean_sim:.3f} < {threshold})",
                      flush=True)

        print(f"Paragraph filtering: {len(collapsed)} -> {len(filtered)} "
              f"groups (threshold={threshold})", flush=True)
        return filtered

    def _align_sentences(self, paragraphs_a, paragraphs_b, collapsed):
        """Align sentences within collapsed paragraph ranges."""
        alignments = []

        for source_indices, target_indices in tqdm.tqdm(collapsed):
            combined_a = " ".join(paragraphs_a[i] for i in source_indices)
            combined_b = " ".join(paragraphs_b[i] for i in target_indices)

            sentence_pairs = fine_align_spans.align_sentence_level(
                combined_a, combined_b,
                model=self.model,
                model_path=self.model_path,
                window=self.thresholds.get("sent_window", 20),
                baseline_threshold=self.thresholds.get("sent_baseline_threshold", 0.70),
                neighbour_margin=self.thresholds.get("sent_neighbour_margin", 0.08),
                fuzzy_threshold=self.thresholds.get("sent_fuzzy_threshold", 0.55),
                min_span_score=self.thresholds.get("sent_min_span_score", 0.60),
            )

            if sentence_pairs:
                alignments.append({
                    "paragraph_A": tuple(source_indices),
                    "paragraph_B": tuple(target_indices),
                    "sentence_pairs": sentence_pairs,
                })

        return sorted(alignments, key=lambda x: (x["paragraph_A"], x["paragraph_B"]))
