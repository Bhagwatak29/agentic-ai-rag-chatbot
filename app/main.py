from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from app.graph import ask_question


# 1. Initialize FastAPI
app = FastAPI(
    title="Agentic AI eBook Chatbot",
    description="A RAG chatbot that answers questions using the Agentic AI eBook.",
    version="1.0.0",
)


# 2. Define the request schema
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question about the Agentic AI eBook",
    )


# 3. Define the source schema
class Source(BaseModel):
    page: Optional[int] = None
    content: str
    relevance_score: Optional[float] = None


# 4. Define the response schema
class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    confidence: Optional[float] = None
    confidence_note: str


# 5. Health check endpoint
@app.get("/")
def home():
    return {
        "message": "Agentic AI eBook Chatbot API is running!",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Agentic AI RAG Chatbot",
    }


# 6. Ask question endpoint
@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    try:
        result = ask_question(request.question)

        # Convert retrieved context into source information
        sources = []

        for doc in result["context"]:
            sources.append(
                Source(
                    page=doc.get("page"),
                    content=doc.get("content", ""),
                    relevance_score=doc.get("final_score"),
                )
            )

        # The cross-encoder score is a relevance score,
        # not a calibrated probability of answer correctness.
        top_score = (
            max(
                (s.relevance_score for s in sources
                 if s.relevance_score is not None),
                default=None,
            )
        )

        return AnswerResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
            confidence=top_score,
            confidence_note=(
                "The confidence field contains the highest retrieval "
                "relevance score, not a probability of answer correctness."
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}",
        )