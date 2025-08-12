# milvus_manager.py
from typing import Dict, List, Optional
import logging, json
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from langchain_milvus import Milvus as LC_Milvus  # vector store wrapper

logger = logging.getLogger(__name__)

BGE_DIM = 1024  # bge-large-en-v1.5

# Only the fields you want (plus text/vector/page_number)
ALLOWED_FIELDS = {
    "text",
    "url", "title", "subtitle", "document_type", "document_number",
    "publication_date", "update_date", "content_hash", "crawl_timestamp",
    "file_size", "lang", "super_category", "crawl_session",
    "top_related", "bottom_related", "themes", "entities", "keywords",
    "page_number",
    "vector",  # (not passed directly; created by vector store)
}

SCHEMA_LIMITS = {
    "url": 2500, "title": 2500, "subtitle": 2500, "document_type": 100,
    "document_number": 100, "publication_date": 50, "update_date": 50,
    "content_hash": 100, "crawl_timestamp": 50, "lang": 10, "super_category": 100,
    "crawl_session": 50,
}

def _clamp(s, n):
    s = "" if s is None else str(s)
    return s[:n]

def build_milvus_metadata_from_crawl(crawl_meta: dict, page_number: int | None) -> dict:
    """Build exactly the metadata you want at top-level, plus page_number."""
    md = {
        "url":             _clamp(crawl_meta.get("url", ""), SCHEMA_LIMITS["url"]),
        "title":           _clamp(crawl_meta.get("title", ""), SCHEMA_LIMITS["title"]),
        "subtitle":        _clamp(crawl_meta.get("subtitle", ""), SCHEMA_LIMITS["subtitle"]),
        "document_type":   _clamp(crawl_meta.get("document_type", ""), SCHEMA_LIMITS["document_type"]),
        "document_number": _clamp(crawl_meta.get("document_number", ""), SCHEMA_LIMITS["document_number"]),
        "publication_date": _clamp(crawl_meta.get("publication_date") or "", SCHEMA_LIMITS["publication_date"]),
        "update_date":      _clamp(crawl_meta.get("update_date") or "", SCHEMA_LIMITS["update_date"]),
        "content_hash":     _clamp(crawl_meta.get("content_hash", ""), SCHEMA_LIMITS["content_hash"]),
        "crawl_timestamp":  _clamp(crawl_meta.get("crawl_timestamp", ""), SCHEMA_LIMITS["crawl_timestamp"]),
        "file_size":        int(crawl_meta.get("file_size") or 0),
        "lang":            _clamp(crawl_meta.get("lang", ""), SCHEMA_LIMITS["lang"]),
        "super_category":  _clamp(crawl_meta.get("super_category", ""), SCHEMA_LIMITS["super_category"]),
        "crawl_session":   _clamp(crawl_meta.get("crawl_session", ""), SCHEMA_LIMITS["crawl_session"]),
        # arrays/dicts as JSON fields:
        "top_related":     crawl_meta.get("top_related", []),
        "bottom_related":  crawl_meta.get("bottom_related", []),
        "themes":          crawl_meta.get("themes", []),
        "entities":        crawl_meta.get("entities", []),
        "keywords":        crawl_meta.get("keywords", []),
    }
    if page_number is not None:
        try:
            md["page_number"] = int(page_number)
        except Exception:
            md["page_number"] = None
    return md

class MilvusManager:
    def __init__(self, connection_args: Dict, collection_name: str, host: str, port: str):
        self.collection_name = collection_name
        self.uri = f"tcp://{host}:{int(port)}"
        self.connection_args = {"uri": self.uri}
        self.collection: Optional[Collection] = None
        self.vector_store: Optional[LC_Milvus] = None
        self._connect()

    def _connect(self):
        connections.connect(alias="default", uri=self.uri)
        logger.info(f"Connected to Milvus at {self.uri}")

    def create_flattened_schema(self, vector_dim: int = BGE_DIM) -> CollectionSchema:
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),

            # content + vector
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),

            # your scalar fields
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=SCHEMA_LIMITS["url"]),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=SCHEMA_LIMITS["title"]),
            FieldSchema(name="subtitle", dtype=DataType.VARCHAR, max_length=SCHEMA_LIMITS["subtitle"]),
            FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=SCHEMA_LIMITS["document_type"]),
            FieldSchema(name="document_number", dtype=DataType.VARCHAR, max_length=SCHEMA_LIMITS["document_number"]),
            FieldSchema(name="publication_date", dtype=DataType.VARCHAR, max_length=SCHEMA_LIMITS["publication_date"]),
            FieldSchema(name="update_date", dtype=DataType.VARCHAR, max_length=SCHEMA_LIMITS["update_date"]),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=SCHEMA_LIMITS["content_hash"]),
            FieldSchema(name="crawl_timestamp", dtype=DataType.VARCHAR, max_length=SCHEMA_LIMITS["crawl_timestamp"]),
            FieldSchema(name="file_size", dtype=DataType.INT64),
            FieldSchema(name="lang", dtype=DataType.VARCHAR, max_length=SCHEMA_LIMITS["lang"]),
            FieldSchema(name="super_category", dtype=DataType.VARCHAR, max_length=SCHEMA_LIMITS["super_category"]),
            FieldSchema(name="crawl_session", dtype=DataType.VARCHAR, max_length=SCHEMA_LIMITS["crawl_session"]),
            FieldSchema(name="page_number", dtype=DataType.INT64),

            # your arrays as JSON fields
            FieldSchema(name="top_related", dtype=DataType.JSON),
            FieldSchema(name="bottom_related", dtype=DataType.JSON),
            FieldSchema(name="themes", dtype=DataType.JSON),
            FieldSchema(name="entities", dtype=DataType.JSON),
            FieldSchema(name="keywords", dtype=DataType.JSON),
        ]
        return CollectionSchema(
            fields=fields,
            description="CSSF docs: only crawl metadata + page_number + JSON arrays + vector",
            enable_dynamic_field=False,  # keep tight
        )

    def create_collection_with_schema(self, embedding_provider, vector_dim: int = BGE_DIM, force_recreate: bool = False):
        if utility.has_collection(self.collection_name):
            if force_recreate:
                utility.drop_collection(self.collection_name)
            else:
                logger.info(f"Collection exists: {self.collection_name}")
        if not utility.has_collection(self.collection_name):
            schema = self.create_flattened_schema(vector_dim)
            self.collection = Collection(name=self.collection_name, schema=schema, using="default")
            logger.info(f"Created collection: {self.collection_name}")
            self._create_indexes()
            self.collection.load()
        else:
            self.collection = Collection(self.collection_name)
            self.collection.load()

        # bind vector store to existing collection
        self.vector_store = LC_Milvus(
            collection_name=self.collection_name,
            embedding_function=embedding_provider,
            connection_args=self.connection_args,
            auto_id=True,
            text_field="text",
            vector_field="vector",
        )
        logger.info(f"Vector store bound to {self.collection_name}")

    def _create_indexes(self):
        try:
            for field in ("url", "title", "document_type", "lang", "content_hash"):
                try:
                    self.collection.create_index(field_name=field, index_params={"index_type": "TRIE", "params": {}})
                except Exception as e:
                    logger.warning(f"Index on {field} skipped: {e}")

            self.collection.create_index(
                field_name="vector",
                index_params={"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}}
            )
            logger.info("Created HNSW(IP) index on vector")
        except Exception as e:
            logger.error(f"Index creation failed: {e}", exc_info=True)

    def _sanitize_metadata(self, md: dict) -> dict:
        """Strict whitelist + types."""
        md = {k: v for k, v in dict(md).items() if k in ALLOWED_FIELDS}

        # scalars
        if "file_size" in md:
            try: md["file_size"] = int(md.get("file_size") or 0)
            except Exception: md["file_size"] = 0
        if "page_number" in md and md["page_number"] is not None:
            try: md["page_number"] = int(md["page_number"])
            except Exception: md["page_number"] = None

        # clamp strings
        for k, lim in SCHEMA_LIMITS.items():
            if k in md and md[k] is not None:
                md[k] = _clamp(md[k], lim)

        # JSON fields: ensure dict/list or coerce from str
        for jf in ("top_related", "bottom_related", "themes", "entities", "keywords"):
            if jf in md:
                v = md[jf]
                if isinstance(v, str):
                    try: md[jf] = json.loads(v)
                    except Exception: md[jf] = {"raw": v}
                elif not isinstance(v, (list, dict)):
                    md[jf] = {"value": v}

        return md

    def add_texts(self, texts: List[str], metadatas: List[Dict]) -> List[str]:
        if not self.vector_store:
            raise RuntimeError("Call create_collection_with_schema() first.")
        metadatas = [self._sanitize_metadata(m) for m in metadatas]
        return self.vector_store.add_texts(texts, metadatas=metadatas)
