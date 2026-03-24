from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class EmbeddingsService:
    """
    Converts text chunks into numerical vectors that capture semantic meaning.
    Using all-mpnet-base-v2 per standard guidelines.
    """
    
    @staticmethod
    def get_embeddings():
        logger.info("Initializing HuggingFaceEmbeddings (all-mpnet-base-v2)...")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
