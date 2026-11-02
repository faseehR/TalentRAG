"""
Vector store layer, behind one interface so retrieval code doesn't
care whether it's talking to Pinecone or a local index.

  - PineconeVectorStore: production. Requires `pip install pinecone`
    and PINECONE_API_KEY. This is the real integration your project
    plan calls for.
  - LocalVectorStore: in-memory cosine-similarity store using numpy.
    No external service needed — this is what runs in this sandbox
    (Pinecone's API isn't reachable from here) and it's also genuinely
    useful for fast local unit tests before you burn Pinecone quota.

Both expose upsert(chunks, vectors) and query(vector, top_k, filter).
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]):
        ...

    @abstractmethod
    def query(self, vector: list[float], top_k: int = 20, filter: dict | None = None) -> list[dict]:
        """Returns list of {id, score, metadata} sorted by descending score."""
        ...


class PineconeVectorStore(BaseVectorStore):
    """
    Production backend.

    Setup (run once):
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        pc.create_index(
            name="talentrag-resumes",
            dimension=<embedding dim, e.g. 1536 for text-embedding-3-small>,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    """

    def __init__(self, index_name: str, namespace: str = "resumes"):
        from pinecone import Pinecone
        import os
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace

    def upsert(self, ids, vectors, metadatas):
        batch_size = 100
        records = list(zip(ids, vectors, metadatas))
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.index.upsert(
                vectors=[
                    {"id": _id, "values": vec, "metadata": meta}
                    for _id, vec, meta in batch
                ],
                namespace=self.namespace,
            )

    def query(self, vector, top_k=20, filter=None):
        res = self.index.query(
            vector=vector,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
            namespace=self.namespace,
        )
        return [
            {"id": m["id"], "score": m["score"], "metadata": m.get("metadata", {})}
            for m in res["matches"]
        ]


class LocalVectorStore(BaseVectorStore):
    """In-memory cosine-similarity store — for local dev/testing without Pinecone."""

    def __init__(self):
        self.ids: list[str] = []
        self._parts: list[np.ndarray] = []   # appended per upsert, merged lazily on query
        self._merged: np.ndarray | None = None
        self._buf: np.ndarray | None = None   # optional preallocated buffer
        self._expected: int | None = None
        self._n = 0
        self.metadatas: list[dict] = []

    def expect(self, total: int):
        """Optional: tell the store how many vectors are coming so it can
        preallocate once instead of copying while growing (saves RAM)."""
        self._expected = total

    @property
    def vectors(self) -> np.ndarray | None:
        if self._buf is not None:
            return self._buf[: self._n]
        # Merge once instead of re-stacking the whole matrix on every upsert
        # (that repeated copying was a big memory spike on small hosts).
        if self._parts:
            parts = ([self._merged] if self._merged is not None else []) + self._parts
            self._merged = np.concatenate(parts) if len(parts) > 1 else parts[0]
            self._parts = []
        return self._merged

    def upsert(self, ids, vectors, metadatas):
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        arr = arr / norms  # pre-normalize for cosine via dot product
        if self._expected and not self._parts and self._merged is None:
            if self._buf is None:
                self._buf = np.empty((self._expected, arr.shape[1]), dtype=np.float32)
            if self._n + len(arr) <= len(self._buf):
                self._buf[self._n:self._n + len(arr)] = arr
                self._n += len(arr)
                self.ids.extend(ids)
                self.metadatas.extend(metadatas)
                return
        self._parts.append(arr)
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)

    def query(self, vector, top_k=20, filter=None):
        vectors = self.vectors
        if vectors is None or len(self.ids) == 0:
            return []
        q = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        sims = vectors @ q  # cosine similarity (both sides normalized)

        candidates = list(range(len(self.ids)))
        if filter:
            candidates = [
                i for i in candidates
                if all(self.metadatas[i].get(k) == v for k, v in filter.items())
            ]
        candidates.sort(key=lambda i: sims[i], reverse=True)
        top = candidates[:top_k]
        return [
            {"id": self.ids[i], "score": float(sims[i]), "metadata": self.metadatas[i]}
            for i in top
        ]


def get_vector_store(backend: str = "local", **kwargs) -> BaseVectorStore:
    if backend == "pinecone":
        return PineconeVectorStore(**kwargs)
    if backend == "local":
        return LocalVectorStore(**kwargs)
    raise ValueError(f"Unknown vector store backend: {backend}")
