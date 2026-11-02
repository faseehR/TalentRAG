"""
Embedding layer. Provides multiple backends behind one interface so
swapping models doesn't touch retrieval/reranking code.

Backends:
  - "openai"   : production — calls OpenAI's embeddings API. Needs
                 OPENAI_API_KEY. Uses text-embedding-3-small by default.
  - "local_tfidf": no external API/model download needed — TF-IDF
                 vectors via scikit-learn. This is what lets the whole
                 retrieval pipeline be built and tested end-to-end
                 right now, without any API keys. Swap to "openai" (or
                 a hosted bge-m3 endpoint) for production — nothing
                 else in the pipeline needs to change.

Note on Pinecone: Pinecone also offers hosted embedding models
(integrated inference), which is often simpler than running your own
embedder — see rag/vector_store.py for that path.
"""

import os
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...


class OpenAIEmbedder(BaseEmbedder):
    """Production backend. Requires `pip install openai` and OPENAI_API_KEY."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Batch to stay under request size limits.
        out = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class GeminiEmbedder(BaseEmbedder):
    """Production backend. Requires `pip install google-genai` and GEMINI_API_KEY.
    Uses 768 dimensions (set your Pinecone index dimension to 768)."""

    def __init__(self, model: str = "gemini-embedding-001", dimensions: int = 768):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dimensions = dimensions

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types
        out = []
        for i in range(0, len(texts), 100):  # API batch limit
            resp = self.client.models.embed_content(
                model=self.model,
                contents=texts[i:i + 100],
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dimensions),
            )
            out.extend([e.values for e in resp.embeddings])
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class LocalTfidfEmbedder(BaseEmbedder):
    """
    Dependency-light stand-in for a real embedding model. Not
    semantically as strong as bge-m3/OpenAI embeddings (no external
    model download required), but implements the exact same interface
    so the rest of the pipeline (vector store, hybrid retrieval,
    reranking, evaluation) can be built and tested right now.

    IMPORTANT: must be fit once on the full corpus before embed_query()
    is used, so the query vector lands in the same vector space as the
    corpus vectors.
    """

    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit(self, corpus_texts: list[str]):
        self.vectorizer.fit(corpus_texts)
        if hasattr(self.vectorizer, "stop_words_"):
            self.vectorizer.stop_words_ = None  # holds every pruned term; wastes RAM
        self._fitted = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("Call .fit(corpus_texts) once before embedding.")
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def get_embedder(backend: str = "local_tfidf", **kwargs) -> BaseEmbedder:
    if backend == "openai":
        return OpenAIEmbedder(**kwargs)
    if backend == "gemini":
        return GeminiEmbedder(**kwargs)
    if backend == "local_tfidf":
        return LocalTfidfEmbedder(**kwargs)
    raise ValueError(f"Unknown embedding backend: {backend}")
