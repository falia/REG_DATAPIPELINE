from __future__ import annotations

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import json
import math

from more_itertools import chunked
from langchain_community.embeddings import SagemakerEndpointEmbeddings
from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain_huggingface import HuggingFaceEmbeddings
import torch


# --- embedding dimension for BAAI/bge-large-en-v1.5 ---
BGE_DIM = 1024


# =========================
# Abstract provider
# =========================
class EmbeddingProvider(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        ...

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        ...


# =========================
# TEI content handler (SageMaker)
# =========================
class TEIContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: List[str], model_kwargs: Dict) -> bytes:
        # TEI payload (supports "inputs": [...]) and optional kwargs passthrough
        payload: Dict[str, Any] = {"inputs": inputs}
        if model_kwargs:
            payload.update(model_kwargs)
        return json.dumps(payload).encode("utf-8")

    @staticmethod
    def _l2norm(v: List[float]) -> List[float]:
        s = math.sqrt(sum(float(x) * float(x) for x in v)) or 1.0
        return [float(x) / s for x in v]

    @staticmethod
    def _to_vectors(obj: Any) -> List[List[float]]:
        # Accept several common TEI shapes:
        # [[...]], {"embeddings":[...]}, [{"embedding":[...]}], {"data":[{"embedding":[...]}]}, {"embedding":[...]}
        if isinstance(obj, dict):
            if "embeddings" in obj:
                return obj["embeddings"]
            if "data" in obj:
                return [row.get("embedding") for row in obj["data"]]
            if "embedding" in obj:
                return [obj["embedding"]]
        if isinstance(obj, list):
            if not obj:
                return []
            if isinstance(obj[0], dict) and "embedding" in obj[0]:
                return [row["embedding"] for row in obj]
            if isinstance(obj[0], (list, tuple)):
                return obj
        raise ValueError(f"Unexpected TEI response: {type(obj)}")

    def transform_output(self, output) -> List[List[float]]:
        data = json.loads(output.read().decode("utf-8"))
        vecs = self._to_vectors(data)
        # client-side unit normalization for inner-product search
        return [self._l2norm(v) for v in vecs]


# =========================
# SageMaker (remote) provider
# =========================
class SageMakerEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        endpoint_name: str = "embedding-endpoint",
        region_name: str = "eu-west-1",
        use_tei: bool = True,
        max_batch_size: int = 8,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.endpoint_name = endpoint_name
        self.region_name = region_name
        self.use_tei = use_tei
        self.max_batch_size = max_batch_size
        self.model_kwargs = model_kwargs or {}

        content_handler = TEIContentHandler()
        self.embeddings = SagemakerEndpointEmbeddings(
            endpoint_name=endpoint_name,
            region_name=region_name,
            content_handler=content_handler,
            # Pass-through to handler via embed_* calls using .kwargs below
        )

    def get_embedding(self, text: str) -> List[float]:
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for batch in chunked(texts, self.max_batch_size):
            # SagemakerEndpointEmbeddings currently forwards only texts; TEIContentHandler
            # already normalizes outputs. If you need to pass model_kwargs (e.g., {"truncate": True}),
            # set self.model_kwargs in __init__ and the handler's transform_input will include them.
            all_embeddings.extend(self.embeddings.embed_documents(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)


# =========================
# Local (optional) provider
# =========================
class LocalEmbeddingProvider(HuggingFaceEmbeddings, EmbeddingProvider):
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        super().__init__(
            model_name=model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # unit-normalize locally
        )

    def get_embedding(self, text: str) -> List[float]:
        return self.embed_query(text)


# =========================
# Embedding service
# =========================
class EmbeddingService:
    def __init__(
        self,
        milvus_config: Optional[Dict] = None,
        use_tei: bool = True,
        vector_dim: int = BGE_DIM,
        use_remote: bool = True,
        **kwargs,
    ):
        # Provider
        if use_remote:
            self.provider: EmbeddingProvider = SageMakerEmbeddingProvider(use_tei=use_tei, **kwargs)
        else:
            self.provider = LocalEmbeddingProvider(**kwargs)

        # Milvus (optional)
        self.milvus = None
        if milvus_config:
            # Make sure this path points to your updated manager
            from embeddings.milvus_provider.milvus_provider import MilvusManager

            host = milvus_config.get("host", "localhost")
            port = milvus_config.get("port", "19530")
            collection_name = milvus_config.get("collection_name", "embeddings")
            connection_args = milvus_config.get("connection_args", {"host": host, "port": port})

            self.milvus = MilvusManager(
                connection_args=connection_args,
                collection_name=collection_name,
                host=host,
                port=port,
            )
            # bind collection & vector store using the current provider
            self.milvus.create_collection_with_schema(
                embedding_provider=self.provider,
                vector_dim=vector_dim,
                force_recreate=False,
            )

    # -------- basic embedding helpers --------
    def create_embedding(self, text: str) -> List[float]:
        return self.provider.get_embedding(text)

    # -------- Milvus helpers --------
    def add_text_to_store(self, text: str, metadata: Dict | None = None) -> Dict:
        if not self.milvus:
            raise RuntimeError("Milvus not configured")
        ids = self.milvus.add_texts([text], [metadata or {}])
        return {"text": text, "milvus_ids": ids, "saved_to_milvus": True, "count": 1}

    def add_texts_to_store(self, texts: List[str], metadatas: List[Dict] | None = None) -> Dict:
        if not self.milvus:
            raise RuntimeError("Milvus not configured")
        if not texts:
            return {"texts": [], "milvus_ids": [], "saved_to_milvus": False, "count": 0}
        metadatas = metadatas or [{}] * len(texts)
        ids = self.milvus.add_texts(texts, metadatas)
        return {"texts": texts, "milvus_ids": ids, "saved_to_milvus": True, "count": len(texts)}

    def search_similar_texts(self, query_text: str, top_k: int = 5, with_scores: bool = False) -> List[Dict]:
        if not self.milvus:
            raise RuntimeError("Milvus not configured")
        return (
            self.milvus.similarity_search_with_score(query_text, k=top_k)
            if with_scores
            else self.milvus.similarity_search(query_text, k=top_k)
        )

    def switch_provider(self, use_remote: bool, use_tei: bool = True, vector_dim: int = BGE_DIM, **kwargs):
        # swap provider
        if use_remote:
            self.provider = SageMakerEmbeddingProvider(use_tei=use_tei, **kwargs)
        else:
            self.provider = LocalEmbeddingProvider(**kwargs)

        # re-bind vector store to the same collection with the new embedding fn
        if self.milvus:
            self.milvus.create_collection_with_schema(
                embedding_provider=self.provider,
                vector_dim=vector_dim,
                force_recreate=False,
            )

    def setup_milvus(
        self,
        host: str = "localhost",
        port: str = "19530",
        connection_args: Dict | None = None,
        collection_name: str = "embeddings",
        vector_dim: int = BGE_DIM,
    ):
        from milvus_provider.milvus_provider import MilvusManager

        connection_args = connection_args or {"host": host, "port": port}
        self.milvus = MilvusManager(
            connection_args=connection_args,
            collection_name=collection_name,
            host=host,
            port=port,
        )
        self.milvus.create_collection_with_schema(
            embedding_provider=self.provider,
            vector_dim=vector_dim,
            force_recreate=False,
        )
