import numpy as np
from .filters import is_valid_sentence_pair
from ..encode.embed_sentences import get_sentence_embeddings
from ..utils.sentence_split import sentence_splitter
from sentence_transformers import util

def cosine_sim(a, b):
	return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)


def fast_monotonic_window_alignment(
    emb_A, emb_B, sentences_A, sentences_B,
    window=20, threshold=0.7
):
    """
    Efficient windowed, monotonic sentence alignment.
    Many-to-one is allowed. Output matches greedy_window_alignment.
    """

    nA = len(emb_A)
    nB = len(emb_B)
    pairs = []

    for i in range(nA):
        # Window bounds
        left = max(0, i - window)
        right = min(nB, i + window + 1)

        candidates = list(range(left, right))
        if not candidates:
            continue

        # Vectorized similarity for all candidates
        cand_embs = emb_B[candidates]               # shape (k, d)
        sims = util.cos_sim(emb_A[i], cand_embs)    # shape (1, k)
        sims = sims.cpu().numpy().flatten()

        # Pick highest scoring candidate
        best_local = sims.argmax()
        best_score = float(sims[best_local])
        best_j = candidates[best_local]

        if best_score >= threshold:
            pairs.append((i, best_j, best_score))

    # --- Enforce monotonicity (critical) ---
    monotonic_pairs = []
    last_j = -1

    for i, j, score in sorted(pairs, key=lambda x: (x[0], x[1])):
        if j > last_j:
            monotonic_pairs.append((i, j, score))
            last_j = j

    return monotonic_pairs



def align_sentence_level(
	paragraph_A_text,
	paragraph_B_text,
	model="labse",
	model_path=None,
):
	# split
	sentences_A = sentence_splitter(paragraph_A_text)
	sentences_B = sentence_splitter(paragraph_B_text)

	if not sentences_A or not sentences_B:
		return []

	# embed
	emb_A = get_sentence_embeddings(sentences_A, model=model, model_path=model_path)
	emb_B = get_sentence_embeddings(sentences_B, model=model, model_path=model_path)

	# greedy alignment
	pairs = fast_monotonic_window_alignment(emb_A, emb_B, sentences_A, sentences_B)

	# fallback drift correction
	if len(pairs) == 0:
		pairs = fast_monotonic_window_alignment(
			emb_A, emb_B, sentences_A, sentences_B,
			window=15, threshold=0.5
		)

	return [
		{
			"src_idx": i,
			"tgt_idx": j,
			"src_text": sentences_A[i],
			"tgt_text": sentences_B[j],
			"score": float(score)
		}
		for (i, j, score) in pairs
	]
