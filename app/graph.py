from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv
import os

from app.retriever import retrieve_documents

load_dotenv()


# 1. Define the state shared between LangGraph nodes
class RAGState(TypedDict):
    question: str
    context: List[Dict[str, Any]]
    answer: str


# 2. Initialize the Groq LLM
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


# 3. Retrieve relevant chunks from the eBook
def retrieve_context(state: RAGState) -> dict:
    question = state["question"]

    documents = retrieve_documents(question)

    return {
        "context": documents
    }


# 4. Generate an answer using retrieved context only
def generate_answer(state: RAGState) -> dict:
    question = state["question"]
    documents = state["context"]

    if not documents:
        return {
            "answer": "I could not find relevant information in the Agentic AI eBook."
        }

    context_text = ""

    for i, doc in enumerate(documents, start=1):
        context_text += (
            f"\nSource {i} (Page {doc.get('page', 'Unknown')}):\n"
            f"{doc.get('content', '')}\n"
        )

    system_prompt = """
You are a helpful assistant answering questions about the Agentic AI eBook.

Rules:
1. Answer using only the provided retrieved context.
2. Do not use outside knowledge or invent information.
3. If the context does not contain the answer, clearly say:
   "The provided eBook context does not contain enough information
   to answer this question."
4. Explain the answer clearly and simply.
5. Include the relevant eBook page numbers in your answer.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Retrieved context:\n{context_text}"
            )
        ),
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content
    }


# 5. Build the LangGraph workflow
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_context)
workflow.add_node("generate", generate_answer)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_graph = workflow.compile()


# 6. Function to invoke the chatbot
def ask_question(question: str) -> dict:
    result = rag_graph.invoke({
        "question": question,
        "context": [],
        "answer": "",
    })

    return {
        "question": result["question"],
        "answer": result["answer"],
        "context": result["context"],
    }


# 7. Run directly from the terminal
if __name__ == "__main__":
    while True:
        question = input(
            "\nAsk a question about the Agentic AI eBook (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        result = ask_question(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        for doc in result["context"]:
            print(
                f"Page {doc.get('page', 'Unknown')} "
                f"| Score: {doc.get('final_score', 'N/A')}"
            )