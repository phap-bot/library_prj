import React, { useState } from 'react';
import { chatApi } from '../../services/chatbotApi';
import './AIChatbot.css'; // Sẽ update style sau để pro-max hơn

const AIChatbot = () => {
    const [messages, setMessages] = useState([
        { role: 'ai', content: 'Xin chào, tôi là trợ lý tri thức RAG. Bạn cần hỏi gì về tài liệu?' }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleSend = async () => {
        if (!input.trim()) return;
        
        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);
        
        try {
            // Simulated delay for RAG pipeline (Retrieval -> Generation)
            // const response = await chatApi.sendMessage(input);
            // setMessages(prev => [...prev, { role: 'ai', content: response.answer }]);
            
            setTimeout(() => {
                setMessages(prev => [...prev, { role: 'ai', content: "\u26A1 Hệ thống RAG Backend đang hoàn thiện Model!" }]);
                setIsLoading(false);
            }, 1000);
            
        } catch (error) {
            setMessages(prev => [...prev, { role: 'error', content: 'Lỗi kết nối Server' }]);
            setIsLoading(false);
        }
    };

    return (
        <div className="rag-chatbot-container glassmorphism-ui">
            <div className="rag-chat-header">
                <h2>Cố Vấn AI Tri Thức</h2>
                <span className="status-indicator online"></span>
            </div>
            
            <div className="rag-chat-messages">
                {messages.map((m, i) => (
                    <div key={i} className={`rag-message-bubble ${m.role}`}>
                        {m.content}
                    </div>
                ))}
                {isLoading && (
                    <div className="rag-message-bubble ai typing">
                        <span className="dot"></span><span className="dot"></span><span className="dot"></span>
                    </div>
                )}
            </div>
            
            <div className="rag-chat-input-area">
                <input 
                    type="text" 
                    value={input} 
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleSend()}
                    placeholder="Đặt câu hỏi từ tài liệu RAG..."
                    disabled={isLoading}
                />
                <button 
                    onClick={handleSend}
                    disabled={isLoading || !input.trim()}
                >
                    Gửi
                </button>
            </div>
        </div>
    );
};

export default AIChatbot;
