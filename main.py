from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
Voce e um assistente util que ajuda os usuarios a responder perguntas com base no contexto fornecido.

Contexto: {context}

Pergunta: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n------------------------")
    question = input("Digite sua pergunta (ou 'q' para encerrar): ")
    print("\n\n")
    if question == 'q':
        break

    docs = retriever.invoke(question)
    if not docs:
        context_text = ""
    else:
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""
            prefix = f"Documento: {source}\n" if source else ""
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            context_parts.append(prefix + content.strip())
        context_text = "\n\n".join(context_parts)

    print(f"Context size: {len(context_text)} caracteres")
    result = chain.invoke({"context": context_text, "question": question})
    print(result)