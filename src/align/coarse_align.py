import numpy as np
from fastdtw import fastdtw
from dtaidistance import dtw

def cosine_distance(a, b):
	return 1 - np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)


def align_paragraphs(emb_A, emb_B, mode="approx"):
	"""
	Returns (path, normalized_cost, dist_matrix):
	  - path: list of (i, j) pairs mapping paragraphs A[i] <-> B[j].
	  - normalized_cost: average cosine distance per alignment step (0 = perfect, 1 = orthogonal).
	  - dist_matrix: pairwise cosine distance matrix (exact mode) or None (approx mode).
	mode: "approx" uses fastdtw, "exact" uses quadratic DTW.
	"""
	if mode == "approx":
		total_cost, path = fastdtw(emb_A, emb_B, dist=cosine_distance)
		norm_cost = total_cost / max(len(path), 1)
		return path, norm_cost, None

	elif mode == "exact":
		# Pairwise cosine distance matrix

		dist_matrix = 1 - np.dot(emb_A, emb_B.T) / (
			np.linalg.norm(emb_A, axis=1)[:, None] *
			np.linalg.norm(emb_B, axis=1)[None, :] + 1e-9
		)

		total_cost, path = dtw_from_cost_matrix(dist_matrix, window=None)
		norm_cost = total_cost / max(len(path), 1)
		return path, norm_cost, dist_matrix

	else:
		raise ValueError(f"Unknown DTW mode: {mode}")

import numpy as np

def dtw_from_cost_matrix(dist_matrix, window=None):
	"""
	Classic exact DTW on a precomputed cost matrix D of shape (N, M).

	Args:
		dist_matrix: 2-D numpy array, D[i,j] = cost of aligning A[i] with B[j]
		window: optional integer w; if provided, only compute cells with |i - j| <= w

	Returns:
		total_cost: float (cumulative cost of best alignment)
		path: list of (i, j) tuples from start -> end (both inclusive)
	"""
	D = np.asarray(dist_matrix, dtype=float)
	N, M = D.shape

	# Large number for initialization
	INF = 1e12

	# Create accumulated cost matrix
	acc = np.full((N, M), INF, dtype=float)

	# Helper to check window constraint
	if window is None:
		def in_window(i, j): return True
	else:
		def in_window(i, j): return abs(i - j) <= window

	# initialize (0,0)
	if in_window(0, 0):
		acc[0, 0] = D[0, 0]
	else:
		raise ValueError("Window excludes (0,0)")

	# first column
	for i in range(1, N):
		j = 0
		if in_window(i, j):
			acc[i, j] = D[i, j] + acc[i-1, j]

	# first row
	for j in range(1, M):
		i = 0
		if in_window(i, j):
			acc[i, j] = D[i, j] + acc[i, j-1]

	# fill the rest
	for i in range(1, N):
		# small optimization: iterate j over feasible window only
		j_start = 1
		j_end = M
		if window is not None:
			j_start = max(1, i - window)
			j_end = min(M, i + window + 1)
		for j in range(j_start, j_end):
			if not in_window(i, j):
				continue
			acc[i, j] = D[i, j] + min(acc[i-1, j], acc[i, j-1], acc[i-1, j-1])

	total_cost = acc[N-1, M-1]

	# backtrack to find path
	i, j = N - 1, M - 1
	if not np.isfinite(acc[i, j]):
		# no valid path in window constraints
		return float('inf'), []

	path = [(i, j)]
	while i > 0 or j > 0:
		if i == 0:
			j -= 1
		elif j == 0:
			i -= 1
		else:
			# pick the direction with minimum accumulated cost
			choices = [(acc[i-1, j-1], i-1, j-1),
					   (acc[i-1, j], i-1, j),
					   (acc[i, j-1], i, j-1)]
			min_cost, ni, nj = min(choices, key=lambda x: x[0])
			i, j = ni, nj
		path.append((i, j))

	path.reverse()
	return float(total_cost), path
