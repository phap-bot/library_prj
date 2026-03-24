from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.rag.pipeline import RAGPipeline
import os
import shutil

router = APIRouter(tags=["Chatbot RAG"])

# Khởi tạo Global Pipeline cho Demo RAG
rag_pipeline = RAGPipeline()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    try:
        answer = await rag_pipeline.ask_question(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-docs")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Lưu file tạm thời để xử lý Loader
        temp_dir = "tests/temp_docs"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Lấy định dạng file (pdf, csv)
        ext = file.filename.split('.')[-1].lower()
        
        # Tiến hành Ingestion Pipeline
        chunks_created = rag_pipeline.ingest_document(temp_file_path, doc_type=ext)
        
        # Có thể xóa file tạm nếu không cần lưu trữ gốc
        os.remove(temp_file_path)
        
        return {
            "message": "Tài liệu đã được AI học thành công!", 
            "chunks_created": chunks_created
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
