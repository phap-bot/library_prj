import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging

logger = logging.getLogger(__name__)


def _format_docs(docs):
    """Combine retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


class GeneratorService:
    """
    Generation Layer: Combines context and query into Prompt, uses LLM (Groq) to generate answer.
    """
    
    def __init__(self, retriever):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Set it in the environment before starting the service."
            )

        logger.info("Initializing ChatGroq LLM (gemma2-9b-it)...")
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            api_key=groq_api_key,
        )
        
        prompt = ChatPromptTemplate.from_template(
            "Use the following context to answer the question. "
            "If you don't know the answer, say you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        
        logger.info("Building LCEL RAG chain (stuff mode)...")
        self.rag_chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
    async def generate_response(self, query: str):
        logger.info(f"Generating response for query: {query}")
        return await self.rag_chain.ainvoke(query)
