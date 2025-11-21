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
			preview_b = sentences_B[j][:45]# + "..." if len(sentences_B[j]) > 45 else sentences_B[j]
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

		if debug and i < 3:  # Show first few
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
		print_section("STAGE 2: Baseline Alignment (window=10, threshold=0.60)")
	
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
	# FUZZY GROUP ALIGN
	# =====================================================================
	if debug:
		print_section("STAGE 4: Fuzzy Group Alignment (threshold=0.60)")
	
	fuzzy = fuzzy_group_align(emb_A, emb_B, threshold=0.90)
	
	if debug:
		print(f"Fuzzy group pairs: {len(fuzzy)}")
		show_pairs(fuzzy, sentences_A, sentences_B, "Fuzzy group pairs")

	# =====================================================================
	# MERGE CANDIDATES
	# =====================================================================
	if debug:
		print_section("STAGE 5: Merging Candidates into Scoring Dictionary")
	
	candidate_scores = {}
	
	def add_candidate(i, j, s):
		key = (int(i), int(j))
		cur = candidate_scores.get(key)
		if cur is None or float(s) > cur:
			candidate_scores[key] = float(s)

	for (i, j, s) in baseline:
		add_candidate(i, j, s)
	for (i, j, s) in expanded:
		add_candidate(i, j, s)
	for (i, j, s) in fuzzy:
		add_candidate(i, j, s)

	if debug:
		print(f"Total unique (i,j) pairs in candidate_scores: {len(candidate_scores)}")
		# Show distribution
		scores_list = sorted(candidate_scores.values(), reverse=True)
		if scores_list:
			print(f"  Score range: {min(scores_list):.4f} to {max(scores_list):.4f}")
			print(f"  Mean score: {np.mean(scores_list):.4f}")

	# =====================================================================
	# BUILD ADJACENCY MAPS
	# =====================================================================
	if debug:
		print_section("STAGE 6: Building Adjacency Maps")
	
	nA, nB = len(sentences_A), len(sentences_B)
	src_to_tgts = {i: set() for i in range(nA)}
	tgt_to_srcs = {j: set() for j in range(nB)}
	
	for (i, j), s in candidate_scores.items():
		src_to_tgts[i].add(j)
		tgt_to_srcs[j].add(i)

	if debug:
		# Show which sources have multiple targets (one-to-many)
		multi_targets = {i: tgts for i, tgts in src_to_tgts.items() if len(tgts) > 1}
		print(f"Sources with multiple targets: {len(multi_targets)}")
		for i, tgts in list(multi_targets.items())[:5]:
			print(f"  A[{i}] → {sorted(tgts)}")
		
		# Show which targets have multiple sources (many-to-one)
		multi_sources = {j: srcs for j, srcs in tgt_to_srcs.items() if len(srcs) > 1}
		print(f"Targets with multiple sources: {len(multi_sources)}")
		for j, srcs in list(multi_sources.items())[:5]:
			print(f"  B[{j}] ← {sorted(srcs)}")

	# =====================================================================
	# PASS A: GREEDY BY SOURCE
	# =====================================================================
	if debug:
		print_section("STAGE 7: PASS A - Greedy by Source (one→many)")
	
	used_src = set()
	used_tgt = set()
	groups = []

	for i in range(nA):
		if i in used_src:
			continue
		candidate_tgts = sorted([j for j in src_to_tgts.get(i, set()) if j not in used_tgt])
		if not candidate_tgts:
			continue
		
		blocks = contiguous_blocks(candidate_tgts)
		best_block = None
		best_block_score = -999.0
		
		for (t0, t1) in blocks:
			scores = []
			for j in range(t0, t1+1):
				sc = candidate_scores.get((i, j))
				if sc is None:
					sc = 0.0
				scores.append(sc)
			avg = sum(scores)/len(scores) if scores else 0.0
			adjusted = avg * (len(scores) ** 0.7)
			if adjusted > best_block_score:
				best_block_score = adjusted
				best_block = (t0, t1, avg)
		
		if best_block is not None:
			t0, t1, avg = best_block
			if any(j in used_tgt for j in range(t0, t1+1)):
				continue
			member_pairs = [(i, j, candidate_scores.get((i, j), avg)) for j in range(t0, t1+1)]
			groups.append((i, i, t0, t1, member_pairs))
			used_src.add(i)
			for j in range(t0, t1+1):
				used_tgt.add(j)

	if debug:
		print(f"After PASS A: {len(groups)} groups, {len(used_src)} src used, {len(used_tgt)} tgt used")

	# =====================================================================
	# PASS B: GREEDY BY TARGET
	# =====================================================================
	if debug:
		print_section("STAGE 8: PASS B - Greedy by Target (many→one)")
	
	for j in range(nB):
		if j in used_tgt:
			continue
		candidate_srcs = sorted([i for i in tgt_to_srcs.get(j, set()) if i not in used_src])
		if not candidate_srcs:
			continue
		
		blocks = contiguous_blocks(candidate_srcs)
		best_block = None
		best_block_score = -999.0
		
		for (s0, s1) in blocks:
			scores = []
			for i in range(s0, s1+1):
				sc = candidate_scores.get((i, j))
				if sc is None:
					sc = 0.0
				scores.append(sc)
			avg = sum(scores)/len(scores) if scores else 0.0
			adjusted = avg * (len(scores) ** 0.7)
			if adjusted > best_block_score:
				best_block_score = adjusted
				best_block = (s0, s1, avg)
		
		if best_block is not None:
			s0, s1, avg = best_block
			if any(i in used_src for i in range(s0, s1+1)):
				continue
			member_pairs = [(i, j, candidate_scores.get((i, j), avg)) for i in range(s0, s1+1)]
			groups.append((s0, s1, j, j, member_pairs))
			used_tgt.add(j)
			for i in range(s0, s1+1):
				used_src.add(i)

	if debug:
		print(f"After PASS B: {len(groups)} groups, {len(used_src)} src used, {len(used_tgt)} tgt used")

	# =====================================================================
	# PASS C: LEFTOVER SINGLETONS
	# =====================================================================
	if debug:
		print_section("STAGE 9: PASS C - Leftover Singletons")
	
	leftover_count = 0
	for (i, j), s in sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
		if i in used_src or j in used_tgt:
			continue
		groups.append((i, i, j, j, [(i, j, s)]))
		used_src.add(i)
		used_tgt.add(j)
		leftover_count += 1

	if debug:
		print(f"Added {leftover_count} leftover singleton pairs")
		print(f"Total groups: {len(groups)}")

	# =====================================================================
	# SORT GROUPS
	# =====================================================================
	if debug:
		print_section("STAGE 10: Sorting Groups by Source Order")
	
	groups.sort(key=lambda g: (g[0], g[2]))
	
	if debug:
		print(f"Groups sorted by (src_start, tgt_start)")

	# =====================================================================
	# BUILD ATOMIC LIST
	# =====================================================================
	if debug:
		print_section("STAGE 11: Building Atomic Pairs List")
	
	atomic = []
	for (s0, s1, t0, t1, member_pairs) in groups:
		for (i, j, coarse) in member_pairs:
			try:
				score = exact_cos(i, j, emb_A, emb_B)
				if np.isnan(score):
					score = float(coarse)
			except Exception as e:
				if debug:
					print(f"  Error computing exact_cos({i}, {j}): {e}")
				score = float(coarse)

			atomic.append({
				"src_idx": int(i),
				"tgt_idx": int(j),
				"src_text": sentences_A[i],
				"tgt_text": sentences_B[j],
				"score": float(score),
			})

	if debug:
		print(f"Atomic pairs: {len(atomic)}")
		for item in atomic[:10]:
			print(f"  A[{item['src_idx']}] ↔ B[{item['tgt_idx']}]  score={item['score']:.4f}")

	# =====================================================================
	# CONNECTED COMPONENTS
	# =====================================================================
	if debug:
		print_section("STAGE 12: Connected Components Analysis")
	
	adj = defaultdict(list)

	for item in atomic:
		s = item["src_idx"]
		t = item["tgt_idx"]
		s_node = ("s", s)
		t_node = ("t", t)
		adj[s_node].append(t_node)
		adj[t_node].append(s_node)

	actual_pairs = set()
	for item in atomic:
		actual_pairs.add((item["src_idx"], item["tgt_idx"]))

	visited = set()
	components = []

	for node in adj:
		if node in visited:
			continue
		queue = deque([node])
		visited.add(node)
		comp = [node]

		while queue:
			x = queue.popleft()
			for y in adj[x]:
				if y not in visited:
					visited.add(y)
					queue.append(y)
					comp.append(y)
		components.append(comp)

	if debug:
		print(f"Connected components: {len(components)}")
		for idx, comp in enumerate(components[:10]):
			src_ids = sorted({n[1] for n in comp if n[0] == "s"})
			tgt_ids = sorted({n[1] for n in comp if n[0] == "t"})
			print(f"  Comp {idx}: A{src_ids} ↔ B{tgt_ids}")

	# =====================================================================
	# EXTRACT FINAL GROUPS
	# =====================================================================
	if debug:
		print_section("STAGE 13: Extracting Final Aligned Groups")
	
	final = []

	for comp in components:
		src_ids = sorted({n[1] for n in comp if n[0] == "s"})
		tgt_ids = sorted({n[1] for n in comp if n[0] == "t"})

		comp_items = []
		for item in atomic:
			pair = (item["src_idx"], item["tgt_idx"])
			if item["src_idx"] in src_ids and item["tgt_idx"] in tgt_ids and pair in actual_pairs:
				comp_items.append(item)

		src_texts = [item["src_text"] for item in comp_items]
		tgt_texts = [item["tgt_text"] for item in comp_items]
		scores    = [item["score"] for item in comp_items]

		final.append({
			"src_idx_start": src_ids[0],
			"src_idx_end":   src_ids[-1],
			"tgt_idx_start": tgt_ids[0],
			"tgt_idx_end":   tgt_ids[-1],
			"src_text": " ".join(src_texts),
			"tgt_text": " ".join(tgt_texts),
			"score": float(np.mean(scores)),
		})

	final.sort(key=lambda x: (x["src_idx_start"], x["tgt_idx_start"]))

	if debug:
		show_groups(final, sentences_A, sentences_B, "Final groups (pre-dedup)")

	# =====================================================================
	# DEDUPLICATION
	# =====================================================================
	if debug:
		print_section("STAGE 14: Deduplication")
	
	deduplicated = []

	for item in final:
		src_range = tuple(range(item["src_idx_start"], item["src_idx_end"]))
		tgt_range = tuple(range(item["tgt_idx_start"], item["tgt_idx_end"]))
		
		src_overlap = any(idx in used_src_indices for idx in src_range)
		tgt_overlap = any(idx in used_tgt_indices for idx in tgt_range)
		
		if not src_overlap and not tgt_overlap:
			deduplicated.append(item)
			used_src_indices.update(src_range)
			used_tgt_indices.update(tgt_range)
			if debug:
				print(f"✓ Kept: A[{item['src_idx_start']}:{item['src_idx_end']}] ↔ B[{item['tgt_idx_start']}:{item['tgt_idx_end']}]")
		else:
			if debug:
				print(f"✗ Removed (overlap): A[{item['src_idx_start']}:{item['src_idx_end']}] ↔ B[{item['tgt_idx_start']}:{item['tgt_idx_end']}]")

	if debug:
		print_section("FINAL RESULT")
		show_groups(deduplicated, sentences_A, sentences_B, "Final deduplicated alignments")

	return deduplicated