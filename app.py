# 1. IMPORTS
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os, uvicorn, logging, re, random  # Added random here
from datetime import datetime
import redis
from typing import List, Dict, Optional
import numpy as np
import json

# 2. LOGGING SETUP
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename=f'logs/chatbot_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 3. APP INITIALIZATION
app = FastAPI(title="Chatbot API")

# 4. MIDDLEWARE SETUP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. REDIS SETUP
redis_client = redis.from_url("redis://localhost", encoding="utf-8", decode_responses=True)
FastAPILimiter.init(redis_client)

# 6. DEVICE & MODEL CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Đang sử dụng: {device}")

MODEL_DIR = "training/results/best_model"

# 7. MODEL LOADING
try:
    print("Đang tải model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_cache=False
    )
    model = model.to(device)
    model.eval()
    print("Đã tải model thành công!")
except Exception as e:
    print(f"Lỗi khi tải model: {str(e)}")
    raise RuntimeError(f"Không thể tải model: {str(e)}")

# 8. STATIC FILES SETUP
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 9. UTILITY CLASSES AND FUNCTIONS
class ChatInput(BaseModel):
    user_input: str
    conversation_history: list = []  # Thêm lịch sử hội thoại

def clean_special_tokens(text):
    text = re.sub(r'<extra_id_\d+>', '', text)
    special_tokens = ['<pad>', '<s>', '</s>', '<unk>']
    for token in special_tokens:
        text = text.replace(token, '')
    return text.strip()

def format_conversation_context(history, current_input):
    """Tạo ngữ cảnh từ lịch sử hội thoại"""
    context = ""
    # Lấy 3 lượt hội thoại gần nhất để tạo ngữ cảnh
    for turn in history[-3:]:
        context += f"Người dùng: {turn['user']}\nBot: {turn['bot']}\n"
    context += f"Người dùng: {current_input}\n"
    return context

# Thêm hàm kiểm tra câu trả lời có phải là lặp lại câu hỏi không
def is_repetitive_answer(question, answer):
    """Kiểm tra xem câu trả lời có phải là lặp lại câu hỏi không"""
    # Chuẩn hóa câu để so sánh
    norm_question = question.lower().strip()
    norm_answer = answer.lower().strip()
    
    # Loại bỏ các tiền tố "hỏi:" và "trả lời:" để so sánh
    norm_question = norm_question.replace("hỏi:", "").strip()
    norm_answer = norm_answer.replace("trả lời:", "").strip()
    
    # Kiểm tra độ tương đồng
    return norm_answer in norm_question or norm_question in norm_answer

# Thêm danh sách các câu trả lời thay thế
FALLBACK_RESPONSES = [
    "Xin lỗi, tôi chưa hiểu rõ câu hỏi của bạn. Bạn có thể diễn đạt khác được không?",
    "Tôi không chắc chắn về câu trả lời. Bạn có thể giải thích rõ hơn được không?",
    "Câu hỏi của bạn khá thú vị, nhưng tôi cần thêm thông tin để trả lời chính xác.",
    "Bạn có thể cung cấp thêm chi tiết về vấn đề này được không?",
    "Tôi đang gặp khó khăn trong việc hiểu câu hỏi. Bạn có thể đặt câu hỏi theo cách khác không?"
]

# Thêm các hàm utility mới
def get_intent(text: str) -> str:
    """Phân tích ý định của câu hỏi"""
    text = text.lower()
    
    # Định nghĩa các pattern cơ bản
    intents = {
        'greeting': ['xin chào', 'hello', 'hi', 'chào'],
        'farewell': ['tạm biệt', 'bye', 'goodbye'],
        'question': ['là gì', 'như thế nào', 'bao giờ', 'thế nào'],
        'thanks': ['cảm ơn', 'thanks', 'thank you'],
    }
    
    for intent, patterns in intents.items():
        if any(p in text for p in patterns):
            return intent
    return 'other'

def enhance_context(history: List[Dict], current_input: str) -> str:
    """Tạo ngữ cảnh nâng cao cho câu hỏi"""
    context = ""
    
    # Thêm thời gian
    current_hour = datetime.now().hour
    time_context = "buổi sáng" if 5 <= current_hour < 12 else \
                   "buổi chiều" if 12 <= current_hour < 18 else "buổi tối"
    context += f"Thời điểm: {time_context}\n"
    
    # Phân tích ý định
    intent = get_intent(current_input)
    context += f"Ý định: {intent}\n"
    
    # Thêm lịch sử hội thoại
    if history:
        last_exchange = history[-1]
        context += f"Câu hỏi trước: {last_exchange['user']}\n"
        context += f"Câu trả lời trước: {last_exchange['bot']}\n"
    
    # Thêm câu hỏi hiện tại
    context += f"Câu hỏi hiện tại: {current_input}\n"
    
    return context

# 10. API ENDPOINTS
@app.post("/chat")
async def chat_response(chat_input: ChatInput):
    user_text = chat_input.user_input.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Vui lòng nhập câu hỏi")

    try:
        # Get intent first
        intent = get_intent(user_text)
        
        # Handle simple intents immediately
        if intent == 'greeting':
            response = "Xin chào! Tôi có thể giúp gì cho bạn?"
            return {
                "response": response,
                "conversation_history": chat_input.conversation_history + [
                    {"user": user_text, "bot": response}
                ],
                "intent": intent
            }
            
        # Format input simpler
        input_text = f"Hỏi: {user_text}"
        logging.info(f"📝 Processing input: {input_text}")
        
        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                min_length=10,
                num_beams=5,
                no_repeat_ngram_size=2,
                repetition_penalty=1.5,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )

        # Process response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = clean_special_tokens(response)
        response = response.strip()

        # Check response quality
        if not response or len(response) < 5 or is_repetitive_answer(user_text, response):
            if intent == 'farewell':
                response = "Tạm biệt! Hẹn gặp lại bạn sau nhé!"
            elif intent == 'thanks':
                response = "Không có gì! Rất vui được giúp bạn."
            else:
                response = random.choice(FALLBACK_RESPONSES)
        
        # Format response
        response = response.capitalize()
        if not response.endswith(('.', '!', '?')):
            response += '.'

        # Update history
        updated_history = chat_input.conversation_history + [
            {"user": user_text, "bot": response}
        ][-5:]

        return {
            "response": response,
            "conversation_history": updated_history,
            "intent": intent
        }

    except Exception as e:
        logging.error(f"Error in chat_response: {str(e)}")
        return {
            "response": "Xin lỗi, tôi đang gặp sự cố. Bạn vui lòng thử lại sau nhé!",
            "conversation_history": chat_input.conversation_history,
            "intent": "error"
        }

@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_DIR,
        "device": str(device),
        "memory": torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A"
    }

# 11. APP STARTUP
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000, 
        log_level="info"
    )
