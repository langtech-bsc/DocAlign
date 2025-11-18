import numpy as np
from fastdtw import fastdtw
from dtaidistance import dtw

def cosine_distance(a, b):
	return 1 - np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)


def align_paragraphs(emb_A, emb_B, mode="approx"):
	"""
	Returns a DTW path: list of (i, j) pairs mapping paragraphs A[i] <-> B[j].
	mode: "approx" uses fastdtw, "exact" uses quadratic DTW.
	"""
	if mode == "approx":
		_, path = fastdtw(emb_A, emb_B, dist=cosine_distance)
		return path

	elif mode == "exact":
		# Pairwise cosine distance matrix
		dist_matrix = 1 - np.dot(emb_A, emb_B.T) / (
			np.linalg.norm(emb_A, axis=1)[:, None] * np.linalg.norm(emb_B, axis=1)[None, :] + 1e-9
		)

		# Perform exact DTW
		path = dtw.warping_path_from_distance_matrix(dist_matrix)

		return path

	else:
		raise ValueError(f"Unknown DTW mode: {mode}")
