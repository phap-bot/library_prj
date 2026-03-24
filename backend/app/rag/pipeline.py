"""
RAG Pipeline Summary module holding Document Ingestion and Query processing logic.
Pipeline Diagram: Document Loading -> Preprocessing -> Chunking -> Embeddings -> Vector DB -> Retrieval -> Generation

Supports two modes:
1. General mode: Direct LLM responses for library-related questions (no documents needed)
2. RAG mode: Full retrieval-augmented generation when documents are uploaded
"""
import os
from typing import List
import logging
from app.rag.loaders import DocumentLoaderService
from app.rag.preprocessors import Preprocessor
from app.rag.chunkers import ChunkerService
from app.rag.embeddings import EmbeddingsService
from app.rag.vector_store import VectorStoreService
from app.rag.retriever import RetrieverService
from app.rag.generator import GeneratorService

logger = logging.getLogger(__name__)


# System prompt for general library assistant mode
LIBRARY_SYSTEM_PROMPT = """Bạn là trợ lý AI thông minh của thư viện SmartLib - Đại học FPT.

Nhiệm vụ chính:
- Hướng dẫn sinh viên quy trình mượn/trả sách tại Kiosk AI.
- Giải đáp thắc mắc về nội quy, giờ giấc, phí phạt và các dịch vụ thư viện.
- Tư vấn và gợi ý tài liệu học tập dựa trên chuyên ngành (IT, Kinh tế, Thiết kế, Ngôn ngữ).
- Kiểm tra hệ thống tồn kho của sách tại thư viện.

Thông tin thư viện SmartLib:
- Giờ mở cửa: 
  + Thứ 2 - Thứ 6: 7:30 - 21:00 (Phục vụ mượn trả & phòng tự học)
  + Thứ 7: 8:00 - 17:00
  + Chủ nhật & Ngày lễ: Nghỉ
- Hạn mức mượn: 
  + Tối đa 5 cuốn sách/sinh viên.
  + Thời hạn: 14 ngày (có thể gia hạn thêm 7 ngày qua ứng dụng di động hoặc tại Kiosk).
- Phí phạt & Đền bù:
  + Trả muộn: 2.000đ/ngày/cuốn.
  + Mất sách/Hư hỏng nặng: Đền bù 150% giá trị sách theo giá thị trường hiện tại.
  

Quy trình mượn sách tại Kiosk:
1. Sinh viên đứng trước camera để AI xác thực khuôn mặt (Face ID).
2. Đặt các cuốn sách cần mượn lên bàn quét AI (AI tự động nhận diện bìa & barcode).
3. Kiểm tra danh sách sách hiện ra trên màn hình.
4. Bấm "Xác nhận mượn".
5. Hệ thống gửi thông báo và biên lai điện tử về Email sinh viên (@fpt.edu.vn).

Quy trình trả sách:
1. Xác thực khuôn mặt tại Kiosk.
2. Vui lòng điền đúng thông tin và xác nhận thông tin cá nhân là chính xác.
3. Đặt sách cần trả vào khay nhận diện.
4. AI kiểm tra tình trạng sách và ghi nhận trả thành công.
5. Nếu có phí phạt quá hạn, hệ thống sẽ hiển thị mã QR để sinh viên thanh toán qua ví điện tử hoặc trừ vào tài khoản sinh viên.

Nội quy phòng đọc & Khu vực tự học:
- Giữ yên lặng tuyệt đối tại khu vực Silent Zone.
- Không mang đồ ăn có mùi hoặc nước uống không có nắp đậy vào phòng máy.
- Trang phục kín đáo, lịch sự (theo quy định đồng phục của Đại học FPT).
- Sau khi dùng sách tại chỗ, vui lòng đặt lại đúng vị trí trên kệ hoặc xe đẩy sách.

Thông tin liên hệ:
- Hotline: 0763537027
- Email: letanphap6543z@gmail.com
- Địa chỉ: Tầng 1, Tòa tháp Alpha, Đại học FPT AI Campus Quy Nhơn.

Quy tắc trả lời của bạn:
- Luôn sử dụng tiếng Việt thân thiện, chuyên nghiệp nhưng vẫn trẻ trung (phù hợp với sinh viên FPT).
- Nếu sinh viên hỏi về các vấn đề ngoài thư viện, hãy khéo léo dẫn dắt về lại chủ đề thư viện.
- Trả lời rõ ràng theo dạng danh sách (bullet points) cho các quy trình phức tạp.
- Khi không chắc chắn, hãy khuyên sinh viên gặp trực tiếp cán bộ thư viện tại quầy Information Desk.
"""


class RAGPipeline:
    def __init__(self):
        self.vector_store = None
        self.generator_service = None
        self._general_llm = None

    def _get_general_llm(self):
        """Lazy-initialize the general-purpose LLM (no RAG needed)."""
        if self._general_llm is None:
            from langchain_groq import ChatGroq
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. Set it in the environment before starting the service."
                )
            self._general_llm = ChatGroq(
                temperature=0.3,
                model_name="llama-3.3-70b-versatile",
                api_key=groq_api_key,
            )
            logger.info("Initialized general-purpose ChatGroq LLM.")
        return self._general_llm

    def ingest_document(self, file_path: str, doc_type: str = "pdf"):
        logger.info(f"Starting ingestion pipeline for {file_path}")
        
        # 1. Document Loading
        if doc_type == "pdf":
            docs = DocumentLoaderService.load_pdf(file_path)
        elif doc_type == "csv":
            docs = DocumentLoaderService.load_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {doc_type}")
            
        # 2. Preprocessing
        clean_docs = Preprocessor.clean_documents(docs)
        
        # 3. Chunking
        chunker = ChunkerService()
        chunks = chunker.split_documents(clean_docs)
        
        # 4. Embeddings
        embeddings_model = EmbeddingsService.get_embeddings()
        
        # 5. Vector DB
        vs_service = VectorStoreService(embeddings_model)
        self.vector_store = vs_service.create_from_documents(chunks)
        
        # Reset generator so it picks up new retriever
        self.generator_service = None
        
        logger.info("Ingestion pipeline completed.")
        return len(chunks)

    async def ask_question(self, query: str) -> str:
        """
        Answer a question. If documents are loaded, use RAG.
        Otherwise, use direct LLM with library system prompt and Tools.
        """
        if self.vector_store:
            return await self._ask_with_rag(query)
        else:
            return await self._ask_general(query)

    async def _ask_general(self, query: str) -> str:
        """Direct LLM response for general library questions."""
        logger.info(f"General mode: answering query: {query}")
        try:
            from langchain.agents import create_tool_calling_agent, AgentExecutor
            from langchain_core.prompts import ChatPromptTemplate
            from app.rag.tools import LIBRARY_TOOLS

            llm = self._get_general_llm()
            
            # Using ChatPromptTemplate suitable for tool calling
            prompt = ChatPromptTemplate.from_messages([
                ("system", LIBRARY_SYSTEM_PROMPT),
                ("human", "{question}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            agent = create_tool_calling_agent(llm, LIBRARY_TOOLS, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=LIBRARY_TOOLS, verbose=True)
            
            result = await agent_executor.ainvoke({"question": query})
            return result["output"]
        except Exception as e:
            logger.error(f"General LLM error: {e}")
            return "Xin lỗi, tôi đang gặp sự cố kết nối. Vui lòng thử lại sau."

    async def _ask_with_rag(self, query: str) -> str:
        """Full RAG pipeline response when documents are available."""
        logger.info(f"RAG mode: answering query: {query}")
        try:
            # 6. Retrieval
            retriever = RetrieverService.create_retriever(self.vector_store, k=5)
            
            # 7. Generation
            if not self.generator_service:
                self.generator_service = GeneratorService(retriever)
                
            return await self.generator_service.generate_response(query)
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            # Fallback to general mode
            logger.info("Falling back to general mode...")
            return await self._ask_general(query)
