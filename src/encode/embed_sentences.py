import os
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_CACHE = {}

def load_model(name="paraphrase-multilingual-MiniLM-L12-v2", model_path=None):
	if name not in _MODEL_CACHE:
		if model_path:
			_MODEL_CACHE[name] = SentenceTransformer(model_path)
		else:
			_MODEL_CACHE[name] = SentenceTransformer(name)
	return _MODEL_CACHE[name]

def get_sentence_embeddings(sentences, model="paraphrase-multilingual-MiniLM-L12-v2", model_path=None):
	model_obj = load_model(model, model_path)
	return model_obj.encode(sentences, convert_to_numpy=True)
