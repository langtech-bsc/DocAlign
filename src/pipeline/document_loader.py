"""Handle loading documents and managing embeddings cache."""
from pathlib import Path
from src.utils import io
from src.encode import embed_paragraphs


class DocumentPair:
    """Represents a pair of documents to align."""
    
    def __init__(self, path_a, path_b, cache_dir=None):
        self.path_a = Path(path_a)
        self.path_b = Path(path_b)
        self.cache_dir = cache_dir or self.path_a.parent
        
        self.lang_a = self.path_a.stem
        self.lang_b = self.path_b.stem
        
        self._paragraphs_a = None
        self._paragraphs_b = None
        self._emb_a = None
        self._emb_b = None
    
    @classmethod
    def from_directory(cls, directory):
        """Create DocumentPair from directory with exactly 2 JSONL files."""
        directory = Path(directory)
        jsonl_files = list(directory.glob("*.jsonl"))
        
        if len(jsonl_files) != 2:
            raise ValueError(f"Expected 2 JSONL files, found {len(jsonl_files)}")
        
        return cls(jsonl_files[0], jsonl_files[1], cache_dir=directory)
    
    def load_paragraphs(self):
        """Load paragraph texts from JSONL files."""
        if self._paragraphs_a is None:
            doc_a = io.load_jsonl(self.path_a)
            self._paragraphs_a = [x["text"] for x in doc_a]
        
        if self._paragraphs_b is None:
            doc_b = io.load_jsonl(self.path_b)
            self._paragraphs_b = [x["text"] for x in doc_b]
        
        return self._paragraphs_a, self._paragraphs_b
    
    def load_embeddings(self, model, model_path=None):
        """Load or compute embeddings with caching."""
        cache_a = self.cache_dir / f"{self.lang_a}.para.npy"
        cache_b = self.cache_dir / f"{self.lang_b}.para.npy"
        
        # Load paragraphs first
        self.load_paragraphs()
        
        # Load/compute embeddings for A
        if cache_a.exists():
            print(f"Loading cached embeddings from {cache_a}", flush=True)
            self._emb_a = io.load_embeddings(cache_a)
        else:
            print(f"Computing embeddings for {self.lang_a}...", flush=True)
            self._emb_a = embed_paragraphs.get_paragraph_embeddings(
                self._paragraphs_a, model=model, model_path=model_path
            )
            io.save_embeddings(cache_a, self._emb_a)
        
        # Load/compute embeddings for B
        if cache_b.exists():
            print(f"Loading cached embeddings from {cache_b}", flush=True)
            self._emb_b = io.load_embeddings(cache_b)
        else:
            print(f"Computing embeddings for {self.lang_b}...", flush=True)
            self._emb_b = embed_paragraphs.get_paragraph_embeddings(
                self._paragraphs_b, model=model, model_path=model_path
            )
            io.save_embeddings(cache_b, self._emb_b)
        
        return self._emb_a, self._emb_b
    
    @property
    def paragraphs_a(self):
        if self._paragraphs_a is None:
            self.load_paragraphs()
        return self._paragraphs_a
    
    @property
    def paragraphs_b(self):
        if self._paragraphs_b is None:
            self.load_paragraphs()
        return self._paragraphs_b
    
    @property
    def emb_a(self):
        if self._emb_a is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings() first.")
        return self._emb_a
    
    @property
    def emb_b(self):
        if self._emb_b is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings() first.")
        return self._emb_b