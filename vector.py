import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

PDF_FOLDER = "base"
DB_LOCATION = "./chroma_langchain_db"

loader = DirectoryLoader(
    path=PDF_FOLDER,
    glob="*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
)
raw_documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
documents = splitter.split_documents(raw_documents)
ids = [f"pdf-{i}" for i in range(len(documents))]

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

if not os.path.exists(DB_LOCATION):
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        persist_directory=DB_LOCATION,
    )
else:
    vector_store = Chroma(
        persist_directory=DB_LOCATION,
        embedding_function=embeddings,
    )
    if documents:
        vector_store.add_documents(documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})