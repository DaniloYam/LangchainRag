import argparse
import json
import os
from collections import OrderedDict

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

from vector import (
    AVAILABLE_DATASETS,
    DATASET_LABELS,
    DATASET_SOURCES,
    get_all_documents,
    get_retriever,
)

load_dotenv()

PROMPT_TEMPLATE = os.getenv(
    "PROMPT_TEMPLATE",
    (
        "Voce e um assistente util que usa somente o contexto da base selecionada.\n"
        "Use todas as bases juntas apenas se o usuario pedir explicitamente.\n"
        "Responda SEMPRE em JSON seguindo o schema:\n"
        "{{\n"
        '  "answer": "<texto da resposta ou vazio>",\n'
        '  "datasets": ["<nome_dataset_usado>", ...],\n'
        '  "error": "<mensagem de erro ou null>"\n'
        "}}\n"
        "Regras:\n"
        "Se o contexto nao trouxer informacoes suficientes, deixe \"answer\" vazio e descreva o motivo em \"error\".\n"
        "Se \"answer\" for NAO VAZIO entao obrigatoriamente coloque \"error\": null.\n"
        "Nunca invente dados fora do contexto.\n"
        "Mencione todos os datasets realmente usados no array \"datasets\".\n\n"
        "Contexto:\n{context}\n\nPergunta: {question}"
    ),
).replace("\\n", "\n")

CLASSIFIER_PROMPT_TEMPLATE = os.getenv(
    "CLASSIFIER_PROMPT_TEMPLATE",
    (
        "Voce e um agente que escolhe qual dataset utilizar antes da resposta final.\n"
        "Bases disponiveis (slug, nome amigavel e PDFs):\n{datasets}\n\n"
        "Responda SEMPRE em JSON seguindo o schema:\n"
        "{{\n"
        '  "collection": "<slug-ou-all>",\n'
        '  "mode": "rag|summary",\n'
        '  "error": "<mensagem de erro ou null>"\n'
        "}}\n"
        "Regras:\n"
        "Use 'summary' quando o usuario pedir um resumo/visao geral ou quando o recurso exigir o contexto completo do dataset (ex.: 'resumo', 'sumario', 'visao geral', 'overview').\n"
        "Use 'rag' para perguntas especificas que podem ser atendidas por busca semantica (retorne os blocos mais relevantes).\n"
        "Use collection='all' apenas se o usuario pedir explicitamente por todas as bases/arquivos.\n"
        "Nao invente colecoes; escolha um slug listado ou 'all'.\n"
        "Se estiver em duvida, prefira collection='all' e mode='rag'.\n\n"
        "Pergunta do usuario: {question}"
    ),
).replace("\\n", "\n")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT]):
    raise RuntimeError(
        "Variaveis AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY e AZURE_OPENAI_DEPLOYMENT sao obrigatorias."
    )

chat_client = ChatCompletionsClient(
    endpoint=AZURE_OPENAI_ENDPOINT,
    credential=AzureKeyCredential(AZURE_OPENAI_API_KEY),
)

BASE_DATASETS = [dataset for dataset in AVAILABLE_DATASETS if dataset != "all"]
CLASSIFIER_CACHE_LIMIT = 25
CLASSIFIER_CACHE: "OrderedDict[str, dict]" = OrderedDict()


def _load_access_rules(path: str) -> dict:
    if not os.path.exists(path):
        raise RuntimeError(f"Arquivo de regras de acesso '{path}' nao encontrado.")
    try:
        with open(path, "r", encoding="utf-8") as handler:
            payload = json.load(handler)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Arquivo de regras '{path}' com JSON invalido.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Arquivo de regras '{path}' deve conter um objeto JSON na raiz.")
    return payload


parser = argparse.ArgumentParser(description="RAG com controle de acesso por usuario")
parser.add_argument("user", help="Identificador do usuario (ex.: joao, pedro, ana)")
parser.add_argument(
    "--access-file",
    default=os.getenv("ACCESS_RULES_PATH", "access_rules.json"),
    help="Caminho para o arquivo JSON com a lista de datasets permitidos por usuario",
)
args = parser.parse_args()

ACCESS_RULES = _load_access_rules(args.access_file)
CURRENT_USER = args.user.strip().lower()
USER_RULE = ACCESS_RULES.get(CURRENT_USER)
if USER_RULE is None:
    raise RuntimeError(
        f"Usuario '{CURRENT_USER}' sem datasets configurados em '{args.access_file}'."
    )


def _normalize_access_rule(rule):
    if isinstance(rule, dict):
        datasets = rule.get("datasets")
        force_all = bool(rule.get("force_all"))
    elif isinstance(rule, list):
        datasets = rule
        force_all = False
    else:
        raise RuntimeError(
            "Cada regra de acesso deve ser uma lista de datasets ou um objeto com o campo 'datasets'."
        )
    if not isinstance(datasets, list) or not datasets:
        raise RuntimeError("Campo 'datasets' deve ser uma lista nao vazia de slugs.")
    return {
        "datasets": [dataset.strip().lower() for dataset in datasets],
        "force_all": force_all,
    }


NORMALIZED_RULE = _normalize_access_rule(USER_RULE)
ALLOWED_DATASETS = set(NORMALIZED_RULE["datasets"])
FORCE_ALL_DATASETS = NORMALIZED_RULE["force_all"] and "all" in ALLOWED_DATASETS

unknown_datasets = ALLOWED_DATASETS.difference(AVAILABLE_DATASETS)
if unknown_datasets:
    raise RuntimeError(
        f"Datasets desconhecidos para o usuario '{CURRENT_USER}': {sorted(unknown_datasets)}."
    )

USER_DATASET_LIST = tuple(sorted(ALLOWED_DATASETS))
USER_BASE_DATASETS = tuple(dataset for dataset in USER_DATASET_LIST if dataset != "all")
if FORCE_ALL_DATASETS:
    DEFAULT_DATASET = "all"
else:
    DEFAULT_DATASET = USER_BASE_DATASETS[0] if USER_BASE_DATASETS else "all"


def _format_allowed() -> str:
    labels = []
    for dataset in sorted(ALLOWED_DATASETS):
        labels.append(DATASET_LABELS.get(dataset, dataset))
    return ", ".join(labels)


print(
    f"Usuario ativo: {CURRENT_USER} | Bases permitidas: {_format_allowed() or 'nenhuma configurada'}"
)


def _format_prompt(context: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=question)


def _call_model(prompt_text: str) -> str:
    response = chat_client.complete(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    }
                ],
            }
        ],
    )
    if not response.choices:
        return ""
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    if not message or not message.content:
        return ""
    text_blocks = []
    for block in message.content:
        if hasattr(block, "text") and block.text:
            text_blocks.append(block.text)
        elif isinstance(block, dict) and block.get("text"):
            text_blocks.append(str(block["text"]))
        else:
            text_blocks.append(str(block))
    return "".join(part for part in text_blocks if part).strip()

def _build_dataset_guide_text() -> str:
    lines = []
    for dataset in USER_BASE_DATASETS:
        label = DATASET_LABELS.get(dataset, dataset)
        sources = DATASET_SOURCES.get(dataset) or []
        if sources:
            preview = ", ".join(sources[:5])
            if len(sources) > 5:
                preview += ", ..."
            sources_text = f"PDFs: {preview}"
        else:
            sources_text = "PDFs: nao especificados"
        lines.append(f"- {dataset}: {label}. {sources_text}")
    if "all" in ALLOWED_DATASETS:
        lines.append("- all: Todas as bases permitidas para este usuario. Use apenas se explicitamente solicitado.")
    if not lines:
        return "- (nenhuma base cadastrada para este usuario)"
    return "\n".join(lines)


def _extract_classifier_payload(raw_text: str) -> dict:
    if not raw_text:
        return {}
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw_text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return {}
        return {}


def _remember_classification(key: str, value: dict) -> None:
    CLASSIFIER_CACHE[key] = value
    CLASSIFIER_CACHE.move_to_end(key)
    while len(CLASSIFIER_CACHE) > CLASSIFIER_CACHE_LIMIT:
        CLASSIFIER_CACHE.popitem(last=False)


def _get_cached_classification(key: str):
    cached = CLASSIFIER_CACHE.get(key)
    if cached is None:
        return None
    CLASSIFIER_CACHE.move_to_end(key)
    return dict(cached)


def _classify_request(question: str) -> dict:
    normalized = question.strip().lower()
    if not normalized:
        return {"collection": "all", "mode": "rag"}

    cached = _get_cached_classification(normalized)
    if cached:
        return cached

    dataset_guide = _build_dataset_guide_text()
    prompt = CLASSIFIER_PROMPT_TEMPLATE.format(datasets=dataset_guide, question=question.strip())
    raw_response = _call_model(prompt)
    payload = _extract_classifier_payload(raw_response)
    collection = (payload.get("collection") or "all").strip().lower()
    mode = (payload.get("mode") or "rag").strip().lower()

    if collection not in AVAILABLE_DATASETS:
        collection = "all"
    if mode not in {"rag", "summary"}:
        mode = "rag"
    if collection == "all" and mode == "summary":
        mode = "rag"

    if FORCE_ALL_DATASETS:
        collection = "all"
    elif collection not in ALLOWED_DATASETS:
        collection = DEFAULT_DATASET

    result = {"collection": collection, "mode": mode}
    _remember_classification(normalized, result)
    return result


while True:
    print("\n\n------------------------")
    question = input("Digite sua pergunta (ou 'q' para encerrar): ").strip()
    print("\n\n")
    if question == 'q':
        break

    classification = _classify_request(question)
    dataset = classification["collection"]
    use_full_context = classification["mode"] == "summary"

    if use_full_context:
        if dataset == "all":
            docs = []
            target_collections = USER_BASE_DATASETS or BASE_DATASETS
            for name in target_collections:
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
        target_collections = USER_BASE_DATASETS or BASE_DATASETS
        for name in target_collections:
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
    mode_label = "summary" if use_full_context else "rag"
    print(f"Dataset selecionado: {selected_label} | modo: {mode_label}")
    print(f"Tamanho do contexto: {len(context_text)} caracteres")
    prompt_text = _format_prompt(context_text, question)
    result = _call_model(prompt_text)
    print(result)