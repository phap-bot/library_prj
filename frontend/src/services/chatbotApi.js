// src/services/chatbotApi.js
// Service layer connect with RAG FastAPI Endpoints

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const chatApi = {
    // 1. Send query to Rag Pipeline for generation
    sendMessage: async (message) => {
        try {
            const response = await fetch(`${BASE_URL}/api/chatbot/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: message })
            });
            if (!response.ok) throw new Error("API Connection Failed");
            return response.json();
        } catch (error) {
            console.error("Chat API error:", error);
            throw error;
        }
    },
    
    // 2. Upload doc to trigger ingestion pipeline
    uploadDocument: async (file) => {
        const formData = new FormData();
        formData.append("file", file);
        try {
            const response = await fetch(`${BASE_URL}/api/chatbot/upload-docs`, {
                method: 'POST',
                body: formData
            });
            return response.json();
        } catch (error) {
            console.error("Upload API error:", error);
            throw error;
        }
    }
};
