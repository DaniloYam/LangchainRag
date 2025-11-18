import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from vector import (
    AVAILABLE_DATASETS,
    DATASET_LABELS,
    DATASET_SOURCES,
    get_all_documents,
    get_retriever,
)

load_dotenv()

MODEL_NAME = os.getenv("LLM_MODEL", "llama3.2")
PROMPT_TEMPLATE = os.getenv(
    "PROMPT_TEMPLATE",
    "Voce e um assistente util que usa somente o contexto da base selecionada.\n"
    "Use todas as bases juntas apenas se o usuario pedir explicitamente.\n\n"
    "Contexto:\n{context}\n\nPergunta: {question}",
).replace("\\n", "\n")

model = OllamaLLM(model=MODEL_NAME)
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
chain = prompt | model

BASE_DATASETS = [dataset for dataset in AVAILABLE_DATASETS if dataset != "all"]
MANUAL_KEYWORDS = ("manual", "guia")
BOOK_KEYWORDS = ("livro",)
FULL_CONTEXT_KEYWORDS = ("resumo",)


def _find_dataset_for_keywords(keywords, exclude=None):
    exclude = set(exclude or [])
    normalized_keywords = tuple(keyword.lower() for keyword in keywords)
    for dataset in BASE_DATASETS:
        if dataset in exclude:
            continue
        label = DATASET_LABELS.get(dataset, dataset).lower()
        slug = dataset.lower()
        sources_text = " ".join(DATASET_SOURCES.get(dataset, [])).lower()
        haystack = f"{label} {slug} {sources_text}"
        if any(keyword in haystack for keyword in normalized_keywords):
            return dataset
    return None


MANUAL_DATASET = _find_dataset_for_keywords(MANUAL_KEYWORDS) or (BASE_DATASETS[0] if BASE_DATASETS else "all")
BOOK_DATASET_DEFAULT = next((dataset for dataset in BASE_DATASETS if dataset != MANUAL_DATASET), MANUAL_DATASET)


def _pick_dataset(question: str) -> str:
    normalized = question.lower()
    if "todos os arquivos" in normalized:
        return "all"
    if any(keyword in normalized for keyword in MANUAL_KEYWORDS):
        return MANUAL_DATASET or "all"
    if any(keyword in normalized for keyword in BOOK_KEYWORDS):
        candidate = _find_dataset_for_keywords(BOOK_KEYWORDS, exclude={MANUAL_DATASET})
        if candidate:
            return candidate
        return BOOK_DATASET_DEFAULT or "all"
    return "all"


while True:
    print("\n\n------------------------")
    question = input("Digite sua pergunta (ou 'q' para encerrar): ").strip()
    print("\n\n")
    if question == 'q':
        break

    normalized_question = question.lower()
    dataset = _pick_dataset(question)
    use_full_context = any(keyword in normalized_question for keyword in FULL_CONTEXT_KEYWORDS)

    if use_full_context:
        if dataset == "all":
            docs = []
            for name in AVAILABLE_DATASETS:
                if name == "all":
                    continue
                docs.extend(get_all_documents(name))
        else:
            docs = get_all_documents(dataset)
    else:
        retriever = get_retriever(dataset)
        docs = retriever.invoke(question)

    if not docs:
        context_text = ""
    else:
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source_name") or doc.metadata.get("source", "")
            dataset_tag = doc.metadata.get("dataset", dataset)
            label = DATASET_LABELS.get(dataset_tag, dataset_tag)
            prefix = f"Dataset: {label} | Documento: {source}\n"
            content = getattr(doc, "page_content", str(doc)).strip()
            context_parts.append(prefix + content)
        context_text = "\n\n".join(context_parts)

    if dataset == "all" and DATASET_SOURCES:
        source_lines = []
        for name in AVAILABLE_DATASETS:
            if name == "all":
                continue
            sources = DATASET_SOURCES.get(name)
            if not sources:
                continue
            joined = ", ".join(sources)
            label = DATASET_LABELS.get(name, name)
            source_lines.append(f"Dataset '{label}' contem: {joined}")
        if source_lines:
            inventory = "\n".join(source_lines)
            context_text = (
                "Inventario de PDFs por dataset:\n"
                f"{inventory}\n\n"
                + context_text
            )

    selected_label = DATASET_LABELS.get(dataset, dataset)
    print(f"Dataset selecionado: {selected_label}")
    print(f"Tamanho do contexto: {len(context_text)} caracteres")
    #print("----------------------------")
    #print(context_text)
    #print("----------------------------")
    result = chain.invoke({"context": context_text, "question": question})
    print(result)