import json
from loguru import logger
from langchain_core.tools import tool
from sqlalchemy import select, or_, func
from app.database import async_session_maker
from app.models.book import Book, BookStatus
from app.models.student import Student
from app.models.transaction import Transaction, TransactionStatus

@tool
async def search_books(query: str) -> str:
    """Tra cứu sách theo tên, tác giả hoặc thể loại trong thư viện. Sử dụng công cụ này khi sinh viên muốn tìm một cuốn sách."""
    logger.info(f"Tool executed: search_books with query '{query}'")
    try:
        async with async_session_maker() as session:
            stmt = select(Book).where(
                or_(
                    Book.title.ilike(f"%{query}%"),
                    Book.author.ilike(f"%{query}%"),
                    Book.subject_category.ilike(f"%{query}%")
                )
            ).limit(5)
            
            result = await session.execute(stmt)
            books = result.scalars().all()
            
            if not books:
                return f"Không tìm thấy sách nào khớp với từ khóa '{query}'."
            
            response = [f"Tìm thấy {len(books)} kết quả:"]
            for b in books:
                status_vn = "Sẵn sàng mượn" if b.status == BookStatus.AVAILABLE else "Đang được mượn/Không khả dụng"
                author = b.author if b.author else "Khuyết danh"
                response.append(f"- '{b.title}' của {author} (Mã sách: {b.book_id}) - Trạng thái: {status_vn}")
                
            return "\n".join(response)
    except Exception as e:
        logger.error(f"Error in search_books tool: {e}")
        return "Xin lỗi, hệ thống đang gặp lỗi khi tra cứu sách."

@tool
async def check_student_info(student_id: str) -> str:
    """Kiểm tra thông tin sinh viên, bao gồm số lượng sách đang mượn và tiền phạt nếu có. Cần chạy công cụ này khi sinh viên muốn biết họ đang nợ sách gì, còn nợ bao nhiêu tiền, hoặc thông tin cá nhân."""
    logger.info(f"Tool executed: check_student_info for student_id '{student_id}'")
    try:
        async with async_session_maker() as session:
            # Get student info
            stmt = select(Student).where(Student.student_id == student_id)
            result = await session.execute(stmt)
            student = result.scalar_one_or_none()
            
            if not student:
                return f"Không tìm thấy sinh viên với mã '{student_id}' trong hệ thống."
                
            # Get active transactions
            t_stmt = select(Transaction).where(
                Transaction.student_id == student_id,
                Transaction.status.in_([TransactionStatus.ACTIVE, TransactionStatus.OVERDUE])
            )
            t_result = await session.execute(t_stmt)
            active_txs = t_result.scalars().all()
            
            info = [
                f"Thông tin sinh viên:",
                f"- Tên: {student.full_name}",
                f"- Mã SV: {student.student_id}",
                f"- Số sách đang mượn (chưa trả): {len(active_txs)} cuốn",
                f"- Tiền phạt đang nợ: {student.fine_balance:,.0f} VNĐ"
            ]
            
            if active_txs:
                info.append("\nChi tiết sách đang mượn:")
                for tx in active_txs:
                    # Also need book details to show name
                    b_stmt = select(Book).where(Book.book_id == tx.book_id)
                    b_res = await session.execute(b_stmt)
                    book = b_res.scalar_one_or_none()
                    title = book.title if book else "Không rõ"
                    
                    status_str = "Quá hạn!" if tx.status == TransactionStatus.OVERDUE else "Đang mượn"
                    info.append(f"  + Cuốn '{title}' (Mã sách: {tx.book_id}) - Hạn trả: {tx.due_date} [{status_str}]")
                    
            return "\n".join(info)
    except Exception as e:
        logger.error(f"Error in check_student_info tool: {e}")
        return "Xin lỗi, hệ thống đang gặp lỗi khi kiểm tra thông tin sinh viên."

# List of tools to pass to the agent
LIBRARY_TOOLS = [search_books, check_student_info]
