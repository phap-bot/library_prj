from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class ChunkerService:
    """
    Step 3: Text Chunking
    Splits large documents into smaller pieces for precise retrieval.
    Configured to chunk_size=300, chunk_overlap=50 for optimal handling.
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,     # Max characters per chunk
            chunk_overlap=50,   # Overlap between chunks to maintain context
            separators=["\n\n", "\n", ".", " "]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(docs)} chunks")
        return docs
