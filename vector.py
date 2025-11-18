import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_FOLDER = "base"
DB_LOCATION = "./chroma_langchain_db"
EMBED_MODEL = "mxbai-embed-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5

def _slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return value.strip("_") or "default"


def _resolve_dataset(filename: str) -> str:
    slug = _slugify(filename)
    return slug


loader = DirectoryLoader(
    path=PDF_FOLDER,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
)
raw_documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)
documents = splitter.split_documents(raw_documents)

grouped_documents: Dict[str, List] = defaultdict(list)
dataset_sources: Dict[str, set] = defaultdict(set)
dataset_labels: Dict[str, str] = {}
for doc in documents:
    source_path = doc.metadata.get("source") or ""
    filename = os.path.basename(source_path) or "documento.pdf"
    slug = _resolve_dataset(filename)
    doc.metadata["dataset"] = slug
    doc.metadata["source_name"] = filename
    grouped_documents[slug].append(doc)
    dataset_sources[slug].add(filename)
    dataset_labels.setdefault(slug, os.path.splitext(filename)[0])

DATASET_DOCUMENTS: Dict[str, List] = {
    slug: list(docs) for slug, docs in grouped_documents.items()
}

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

BASE_RETRIEVERS: Dict[str, object] = {}

for slug, docs in grouped_documents.items():
    store = Chroma(
        collection_name=slug,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings,
    )
    existing = store.get(include=[])
    if existing and existing.get("ids"):
        store.delete(ids=existing["ids"])
    ids = [f"{slug}-{i}" for i in range(len(docs))]
    if docs:
        store.add_documents(docs, ids=ids)
    BASE_RETRIEVERS[slug] = store.as_retriever(search_kwargs={"k": DEFAULT_TOP_K})


class MultiCollectionRetriever:
    def __init__(self, retrievers: Dict[str, object]):
        self._retrievers = dict(retrievers)

    def invoke(self, query: str):
        results = []
        for slug, retriever_obj in self._retrievers.items():
            docs = retriever_obj.invoke(query)
            for doc in docs:
                doc.metadata.setdefault("dataset", slug)
            results.extend(docs)
        return results

    def set_k(self, k: int):
        for retriever_obj in self._retrievers.values():
            if hasattr(retriever_obj, "search_kwargs"):
                retriever_obj.search_kwargs["k"] = k


COLLECTION_RETRIEVERS: Dict[str, object] = dict(BASE_RETRIEVERS)
COLLECTION_RETRIEVERS["all"] = MultiCollectionRetriever(BASE_RETRIEVERS)
BASE_DATASETS = tuple(sorted(BASE_RETRIEVERS.keys()))
AVAILABLE_DATASETS = BASE_DATASETS + ("all",)

DATASET_SOURCES: Dict[str, List[str]] = {
    dataset: sorted(list(names)) for dataset, names in dataset_sources.items()
}
if dataset_sources:
    all_sources = sorted({name for names in dataset_sources.values() for name in names})
    DATASET_SOURCES["all"] = all_sources

DATASET_LABELS: Dict[str, str] = {
    dataset: dataset_labels.get(dataset, dataset) for dataset in BASE_DATASETS
}

DATASET_LABELS["all"] = "Todas as bases combinadas"


def get_all_documents(dataset: Optional[str] = None) -> List:
    if dataset in (None, "all"):
        combined = []
        for docs in DATASET_DOCUMENTS.values():
            combined.extend(docs)
        return list(combined)
    docs = DATASET_DOCUMENTS.get(dataset)
    if docs is None:
        raise ValueError(f"Dataset '{dataset}' indisponivel.")
    return list(docs)


def get_retriever(dataset: Optional[str] = None, *, k: Optional[int] = None):
    target = dataset or "all"
    retriever_obj = COLLECTION_RETRIEVERS.get(target)
    if retriever_obj is None:
        raise ValueError(f"Dataset '{target}' indisponivel. Opcoes: {AVAILABLE_DATASETS}")
    if k:
        if target == "all" and hasattr(retriever_obj, "set_k"):
            retriever_obj.set_k(k)
        elif hasattr(retriever_obj, "search_kwargs"):
            retriever_obj.search_kwargs["k"] = k
    return retriever_obj
