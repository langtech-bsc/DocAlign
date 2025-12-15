import numpy as np
from collections import defaultdict, deque
from .filters import is_valid_sentence_pair
from ..encode.embed_sentences import get_sentence_embeddings
from ..utils.sentence_split import sentence_splitter
from sentence_transformers import util
import json
from pathlib import Path
import hashlib


# =====================================================================
# EMBEDDING CACHE (optional, for faster testing)
# =====================================================================
class EmbeddingCache:
	"""Simple file-based cache for sentence embeddings."""
	
	def __init__(self, cache_dir=".embed_cache", enabled=True):
		self.cache_dir = Path(cache_dir)
		self.enabled = enabled
		if self.enabled:
			self.cache_dir.mkdir(exist_ok=True)
	
	def _hash_text(self, text):
		"""Create a hash key from text."""
		return hashlib.md5(text.encode()).hexdigest()
	
	def get(self, text, model="labse"):
		"""Retrieve cached embeddings if they exist."""
		if not self.enabled:
			return None
		key = self._hash_text(text)
		cache_file = self.cache_dir / f"{model}_{key}.npy"
		if cache_file.exists():
			return np.load(cache_file)
		return None
	
	def set(self, text, embeddings, model="labse"):
		"""Cache embeddings."""
		if not self.enabled:
			return
		key = self._hash_text(text)
		cache_file = self.cache_dir / f"{model}_{key}.npy"
		np.save(cache_file, embeddings)

_cache = EmbeddingCache(enabled=True)  # Set to False to disable caching


# =====================================================================
# DEBUGGING UTILITIES
# =====================================================================
def print_section(title):
	"""Print a section header."""
	print(f"\n{'='*70}")
	print(f"  {title}")
	print(f"{'='*70}")


def show_sentences(sentences, label, max_show=10):
	"""Pretty-print sentences."""
	print(f"\n{label} ({len(sentences)} total):")
	for idx, sent in enumerate(sentences[:max_show]):
		preview = sent[:60]# + "..." if len(sent) > 60 else sent
		print(f"  [{idx:2d}] {preview}")
	if len(sentences) > max_show:
		print(f"  ... and {len(sentences) - max_show} more")


def show_pairs(pairs, sentences_A, sentences_B, label, max_show=15):
	"""Pretty-print aligned pairs with scores."""
	print(f"\n{label} ({len(pairs)} pairs):")
	for idx, (i, j, score) in enumerate(pairs[:max_show]):
		print(f"  [{idx:2d}] A[{i:2d}] <-> B[{j:2d}]  score={score:.4f}")
		if i < len(sentences_A):
			preview_a = sentences_A[i]#[:45] + "..." if len(sentences_A[i]) > 45 else sentences_A[i]
			print(f"       A: {preview_a}")
		if j < len(sentences_B):
			preview_b = sentences_B[j]#[:45] + "..." if len(sentences_B[j]) > 45 else sentences_B[j]
			print(f"       B: {preview_b}")
	if len(pairs) > max_show:
		print(f"  ... and {len(pairs) - max_show} more pairs")


def show_groups(groups, sentences_A, sentences_B, label):
	"""Pretty-print final groups."""
	print(f"\n{label} ({len(groups)} groups):")
	for idx, group in enumerate(groups):
		src_start, src_end, tgt_start, tgt_end = group["src_idx_start"], group["src_idx_end"], group["tgt_idx_start"], group["tgt_idx_end"]
		print(f"\n  [Group {idx}] A[{src_start}:{src_end+1}] <-> B[{tgt_start}:{tgt_end+1}]  score={group['score']:.4f}")
		print(f"    A: {group['src_text']}")
		print(f"    B: {group['tgt_text']}")


# =====================================================================
# ORIGINAL FUNCTIONS (unchanged)
# =====================================================================
def cosine_sim(a, b):
	return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)


def mean_embed(emb_list):
	if len(emb_list) == 0:
		return None
	return np.mean(np.vstack(emb_list), axis=0)

def extract_fuzzy_spans(emb_A, emb_B, threshold=0.55):
	"""
	Extract span-based fuzzy matches directly.
	Returns matches like A[i] ↔ B[j:j+1] or A[i:i+1] ↔ B[j]
	"""
	nA, nB = len(emb_A), len(emb_B)
	spans = []

	# 1-to-2 matches: One sentence in A matches two in B
	for i in range(nA):
		for j in range(nB - 1):
			combined_B = mean_embed([emb_B[j], emb_B[j+1]])
			sim = cosine_sim(emb_A[i], combined_B)
			if sim >= threshold:
				spans.append({
					'src_span': (i, i),      # Single sentence
					'tgt_span': (j, j+1),    # Two sentences
					'score': sim
				})

	# 2-to-1 matches: Two sentences in A match one in B
	for i in range(nA - 1):
		combined_A = mean_embed([emb_A[i], emb_A[i+1]])
		for j in range(nB):
			sim = cosine_sim(combined_A, emb_B[j])
			if sim >= threshold:
				spans.append({
					'src_span': (i, i+1),    # Two sentences
					'tgt_span': (j, j),      # Single sentence
					'score': sim
				})

	return spans

def expand_neighbours(pairs, emb_A, emb_B, sentences_A, sentences_B,
					  threshold, neighbour_margin=0.1):
	expanded = set(pairs)

	for (i, j, score) in pairs:
		for dj in (-1, 1):
			jj = j + dj
			if 0 <= jj < len(emb_B):
				sim = cosine_sim(emb_A[i], emb_B[jj])
				if sim >= threshold - neighbour_margin:
					expanded.add((i, jj, sim))

		for di in (-1, 1):
			ii = i + di
			if 0 <= ii < len(emb_A):
				sim = cosine_sim(emb_A[ii], emb_B[j])
				if sim >= threshold - neighbour_margin:
					expanded.add((ii, j, sim))

	return list(expanded)


def fuzzy_group_align(emb_A, emb_B, threshold=0.70):
	nA, nB = len(emb_A), len(emb_B)
	results = []

	for i in range(nA):
		for j in range(nB - 1):
			combined_B = mean_embed([emb_B[j], emb_B[j+1]])
			sim = cosine_sim(emb_A[i], combined_B)
			if sim >= threshold:
				results.append((i, j, sim))
				results.append((i, j+1, sim))

	for i in range(nA - 1):
		combined_A = mean_embed([emb_A[i], emb_A[i+1]])
		for j in range(nB):
			sim = cosine_sim(combined_A, emb_B[j])
			if sim >= threshold:
				results.append((i, j, sim))
				results.append((i+1, j, sim))

	return results


def smooth_monotonicity(pairs, local_slack=2):
	pairs_sorted = sorted(pairs, key=lambda x: (x[0], x[1]))
	result = []
	last_j = -1

	for i, j, score in pairs_sorted:
		if j + local_slack >= last_j:
			result.append((i, j, score))
			last_j = max(last_j, j)

	return result


def fast_monotonic_window_alignment(
	emb_A, emb_B, sentences_A, sentences_B,
	window=10, threshold=0.7, debug=False
):
	nA = len(emb_A)
	nB = len(emb_B)
	pairs = []

	for i in range(nA):
		left = max(0, i - window)
		right = min(nB, i + window + 1)
		candidates = list(range(left, right))
		if not candidates:
			continue

		cand_embs = emb_B[candidates]
		sims = util.cos_sim(emb_A[i], cand_embs).cpu().numpy().flatten()

		best_local = sims.argmax()
		best_score = float(sims[best_local])
		best_j = candidates[best_local]

		if debug:  # Show first few
			print(f"  A[{i}]: best_j={best_j}, score={best_score:.4f}, candidates={candidates}")

		if best_score >= threshold:
			pairs.append((i, best_j, best_score))

	# strict monotonicity
	pairs_sorted = sorted(pairs, key=lambda x: (x[0], x[1]))
	monotonic_pairs = []
	last_j = -1
	for i, j, score in pairs_sorted:
		if j > last_j:
			monotonic_pairs.append((i, j, score))
			last_j = j

	return monotonic_pairs


def contiguous_blocks(sorted_idxs):
	"""Helper: contiguous blocks from a sorted list of ints"""
	if not sorted_idxs:
		return []
	blocks = []
	start = sorted_idxs[0]
	last = start
	for x in sorted_idxs[1:]:
		if x == last + 1:
			last = x
		else:
			blocks.append((start, last))
			start = x
			last = x
	blocks.append((start, last))
	return blocks


def exact_cos(i, j, emb_A, emb_B):
	"""Compute exact cosine similarity between two embeddings."""
	v = emb_A[i]
	w = emb_B[j]
	denom = (np.linalg.norm(v) * np.linalg.norm(w) + 1e-9)
	return float(np.dot(v, w) / denom)


# =====================================================================
# MAIN FUNCTION WITH DEBUGGING CHECKPOINTS
# =====================================================================
def align_sentence_level(
	paragraph_A_text,
	paragraph_B_text,
	model="labse",
	model_path=None,
	used_src_indices=None,
	used_tgt_indices=None,
	debug=True,  # NEW: Enable debugging
):
	"""
	Align sentences with extensive debugging checkpoints.
	
	Args:
		debug: If True, prints detailed info at each stage
	"""
	if debug:
		print_section("STAGE 0: Input & Sentence Splitting")
	
	sentences_A = sentence_splitter(paragraph_A_text)
	sentences_B = sentence_splitter(paragraph_B_text)

	if debug:
		show_sentences(sentences_A, "Paragraph A sentences", max_show=8)
		show_sentences(sentences_B, "Paragraph B sentences", max_show=8)

	if not sentences_A or not sentences_B:
		if debug:
			print("ERROR: Empty sentence list!")
		return []

	if used_src_indices is None:
		used_src_indices = set()
	if used_tgt_indices is None:
		used_tgt_indices = set()

	# =====================================================================
	# EMBEDDINGS (with caching)
	# =====================================================================
	if debug:
		print_section("STAGE 1: Computing Embeddings (with caching)")
	
	# Try to get cached embeddings
	text_a_key = "\n".join(sentences_A)
	text_b_key = "\n".join(sentences_B)
	
	emb_A = _cache.get(text_a_key, model)
	if emb_A is None:
		if debug:
			print("Computing embeddings for A (not in cache)...")
		emb_A = get_sentence_embeddings(sentences_A, model=model, model_path=model_path)
		_cache.set(text_a_key, emb_A, model)
	else:
		if debug:
			print("✓ Loaded A embeddings from cache")
	
	emb_B = _cache.get(text_b_key, model)
	if emb_B is None:
		if debug:
			print("Computing embeddings for B (not in cache)...")
		emb_B = get_sentence_embeddings(sentences_B, model=model, model_path=model_path)
		_cache.set(text_b_key, emb_B, model)
	else:
		if debug:
			print("✓ Loaded B embeddings from cache")
	
	if debug:
		print(f"emb_A shape: {emb_A.shape}, emb_B shape: {emb_B.shape}")

	# =====================================================================
	# BASELINE
	# =====================================================================
	if debug:
		print_section("STAGE 2: Baseline Alignment (window=10, threshold=0.70)")
	
	baseline = fast_monotonic_window_alignment(
		emb_A, emb_B, sentences_A, sentences_B,
		window=20, threshold=0.70, debug=debug
	)
	
	if debug:
		show_pairs(baseline, sentences_A, sentences_B, "Baseline pairs")

	if len(baseline) == 0:
		if debug:
			print("  ⚠ Baseline empty")

	# =====================================================================
	# NEIGHBOR EXPANSION
	# =====================================================================
	if debug:
		print_section("STAGE 3: Neighbor Expansion (threshold=0.70, margin=0.08)")
	
	expanded = expand_neighbours(baseline, emb_A, emb_B, sentences_A, sentences_B,
								 threshold=0.70, neighbour_margin=0.08)
	
	if debug:
		print(f"Baseline: {len(baseline)} pairs → Expanded: {len(expanded)} pairs")
		show_pairs(expanded, sentences_A, sentences_B, "Expanded pairs (after neighbor expansion)")

	# =====================================================================
	# STAGE 4: Fuzzy Span Extraction (1-to-2 and 2-to-1)
	# =====================================================================
	if debug:
		print_section("STAGE 4: Fuzzy Span Extraction (span-based)")

	fuzzy_spans = extract_fuzzy_spans(emb_A, emb_B, threshold=0.7)
	if debug:
		print(f"Found {len(fuzzy_spans)} fuzzy span candidates")

	# =====================================================================
	# STAGE 5: Build Unified Span Candidate List
	# =====================================================================
	if debug:
		print_section("STAGE 5: Build Span Candidates")

	span_candidates = []

	# Baseline 1-1
	for (i, j, score) in baseline:
		span_candidates.append({
			'src_span': (i, i),
			'tgt_span': (j, j),
			'score': score,
			'type': 'baseline'
		})

	# Expanded 1-1
	for (i, j, score) in expanded:
		span_candidates.append({
			'src_span': (i, i),
			'tgt_span': (j, j),
			'score': score,
			'type': 'expanded'
		})

	# Spans from fuzzy
	for fs in fuzzy_spans:
		span_candidates.append({
			'src_span': fs['src_span'],
			'tgt_span': fs['tgt_span'],
			'score': fs['score'],
			'type': 'fuzzy'
		})

	if debug:
		total = len(span_candidates)
		base = len([x for x in span_candidates if x['type']=='baseline'])
		expd = len([x for x in span_candidates if x['type']=='expanded'])
		fz   = len([x for x in span_candidates if x['type']=='fuzzy'])
		print(f"Total={total}  baseline={base}  expanded={expd}  fuzzy={fz}")

	# =====================================================================
	# STAGE 6: Greedy Span Selection
	# =====================================================================
	if debug:
		print_section("STAGE 6: Greedy Span Selection")

	span_candidates.sort(key=lambda x: -x['score'])

	used_src = set()
	used_tgt = set()
	selected = []

	MIN_SPAN_SCORE = 0.60

	for cand in span_candidates:
		src_start, src_end = cand['src_span']
		tgt_start, tgt_end = cand['tgt_span']
		score = cand['score']

		if score < MIN_SPAN_SCORE:
			continue

		src_set = set(range(src_start, src_end + 1))
		tgt_set = set(range(tgt_start, tgt_end + 1))

		if src_set & used_src:
			continue
		if tgt_set & used_tgt:
			continue

		selected.append(cand)
		used_src |= src_set
		used_tgt |= tgt_set

		if debug:
			print(f"✓ Selected A[{src_start}:{src_end+1}] ↔ B[{tgt_start}:{tgt_end+1}] "
				  f"score={score:.4f} ({cand['type']})")

	# =====================================================================
	# STAGE 7: Build Final Output (span groups)
	# =====================================================================
	if debug:
		print_section("STAGE 7: Build Final Output")

	final = []

	for cand in selected:
		src_start, src_end = cand['src_span']
		tgt_start, tgt_end = cand['tgt_span']

		src_text = " ".join(sentences_A[src_start:src_end+1])
		tgt_text = " ".join(sentences_B[tgt_start:tgt_end+1])

		final.append({
			"src_idx_start": src_start,
			"src_idx_end": src_end,
			"tgt_idx_start": tgt_start,
			"tgt_idx_end": tgt_end,
			"src_text": src_text,
			"tgt_text": tgt_text,
			"score": float(cand['score'])
		})

	final.sort(key=lambda x: (x["src_idx_start"], x["tgt_idx_start"]))

	if debug:
		show_groups(final, sentences_A, sentences_B, "Final span-based alignments")

	return final
