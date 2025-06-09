# 1. IMPORTS
from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)
import os, uvicorn, logging, re, random, asyncio, json
from datetime import datetime, timezone # Import timezone để đảm bảo thời gian lưu là UTC
import redis.asyncio as redis
from typing import List, Dict, Optional, Union, Any, Callable # Thêm Callable
from pathlib import Path
import numpy as np
from functools import lru_cache

# --- THAY THẾ CÁC IMPORTS CỦA POSTGRESQL/SQLALCHEMY BẰNG MONGODB ---
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId # Để xử lý ObjectId của MongoDB nếu cần
from pydantic_core import core_schema, PydanticCustomError # THÊM DÒNG NÀY CHO PYDANTIC V2 và PydanticCustomError

# Thêm vào phần imports
templates = Jinja2Templates(directory="templates")

# 2. CONFIG CLASS
class ChatbotConfig:
    # Model parameters
    MAX_HISTORY = 5
    MAX_LENGTH = 256
    MIN_LENGTH = 10
    TEMPERATURE = 0.7
    TOP_K = 50
    TOP_P = 0.9
    NUM_BEAMS = 5
    NO_REPEAT_NGRAM_SIZE = 2
    REPETITION_PENALTY = 1.5
    LENGTH_PENALTY = 1.0
    
    # Training parameters
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 100
    MAX_GRAD_NORM = 1.0
    
    # Redis config
    REDIS_URL = "redis://localhost"
    REDIS_TIMEOUT = 5
    CACHE_TTL = 3600  # 1 hour
    
    # API rate limits
    RATE_LIMIT_TIMES = 10
    RATE_LIMIT_MINUTES = 1

    # --- THÊM CẤU HÌNH MONGODB ---
    MONGO_DB_URL = "mongodb://localhost:27017/"
    MONGO_DB_NAME = "chatbot_db" # Tên database của bạn trong MongoDB
    CONVERSATIONS_COLLECTION_NAME = "conversations" # Tên collection
    # --- KẾT THÚC THÊM CẤU HÌNH MONGODB ---

# Global vars for DB
# --- THAY THẾ GLOBAL VARS CHO DB ---
mongo_client: Optional[AsyncIOMotorClient] = None
mongo_db = None
conversations_collection = None
# --- KẾT THÚC THAY THẾ ---

# Fallback responses for the chatbot in case of model loading or generation errors
FALLBACK_RESPONSES = [
    "Xin lỗi, tôi đang gặp vấn đề. Vui lòng thử lại sau.",
    "Tôi không thể phản hồi lúc này. Mong bạn thông cảm.",
    "Có vẻ như có lỗi xảy ra. Hãy thử một câu hỏi khác nhé!",
    "Tôi đang học hỏi. Vui lòng thử lại sau ít phút."
]

# 3. LOGGING SETUP
def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        filename=log_dir / f'chatbot_{datetime.now().strftime("%Y%m%d")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

setup_logging()

# 4. PYDANTIC MODELS
class ChatInput(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=500)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    language: str = Field(default="vi")
    context: Optional[Dict[str, str]] = None

    @validator('user_input')
    def validate_input(cls, v):
        if not v.strip():
            raise ValueError('Input không được để trống')
        return v.strip()

# --- THÊM Pydantic Model cho Feedback Input (để xử lý ObjectId) ---
class FeedbackInput(BaseModel):
    # conversation_id sẽ là string vì ObjectId là string trong JSON
    conversation_id: str = Field(..., description="ID của cuộc hội thoại cần đánh giá.")
    score: int = Field(..., ge=-1, le=1, description="Điểm đánh giá: 1 (tốt), 0 (trung lập), -1 (kém).")
    comment: Optional[str] = Field(None, max_length=500, description="Bình luận thêm (tùy chọn).")

# Helper để chuyển ObjectId sang string khi trả về JSON và tương thích Pydantic v2
class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Callable[[Any], core_schema.CoreSchema]) -> core_schema.CoreSchema:
        def validate_from_anything(value: Any) -> ObjectId:
            if isinstance(value, ObjectId):
                return value
            if isinstance(value, str):
                if not ObjectId.is_valid(value):
                    raise PydanticCustomError('object_id', 'Invalid ObjectId string')
                return ObjectId(value)
            raise PydanticCustomError('object_id', 'ObjectId must be a string or ObjectId instance')

        # Validator cho đầu vào
        validator_schema = core_schema.no_info_plain_validator_function(validate_from_anything)

        # Serializer cho đầu ra
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),  # ❌ Đã bỏ extra={}
            python_schema=validator_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(lambda o: str(o))
        )



class ConversationOut(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: Optional[str]
    timestamp: datetime
    user_input: str
    bot_response: str
    intent: str
    context: Dict[str, Any]
    feedback_score: Optional[int]
    feedback_timestamp: Optional[datetime]
    is_good_data: bool

    class Config:
        populate_by_name = True # Đã thay thế allow_population_by_field_name cho Pydantic v2
        arbitrary_types_allowed = True # Cho phép Pydantic xử lý kiểu ObjectId
        json_encoders = {ObjectId: str} # Cách chuyển ObjectId sang string khi serialize
# --- KẾT THÚC THÊM Pydantic Model ---


# 5. APP INITIALIZATION
app = FastAPI(
    title="Chatbot API",
    description="API cho Chatbot AI sử dụng transformer",
    version="1.0.0"
)

# 6. MIDDLEWARE SETUP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thêm sau phần MIDDLEWARE SETUP
app.mount("/static", StaticFiles(directory="static"), name="static")

# 7. REDIS SETUP
redis_client = None
USE_REDIS = False  # Thêm flag để kiểm soát việc sử dụng Redis, đặt False nếu không dùng

@app.on_event("startup")
async def startup_event():
    global redis_client, mongo_client, mongo_db, conversations_collection # Thêm các biến MongoDB vào global
    
    # ... (phần Redis setup hiện có) ...
    if USE_REDIS: # Giữ nguyên phần Redis
        try:
            redis_client = redis.from_url(
                ChatbotConfig.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=ChatbotConfig.REDIS_TIMEOUT,
                retry_on_timeout=True
            )
            await FastAPILimiter.init(redis_client)
            logging.info("Redis connection established")
        except Exception as e:
            logging.error(f"Redis connection failed: {str(e)}")
            redis_client = None

    # --- THAY THẾ PHẦN DATABASE SETUP CHO MONGODB ---
    logging.info("Initializing MongoDB connection...")
    try:
        mongo_client = AsyncIOMotorClient(ChatbotConfig.MONGO_DB_URL)
        mongo_db = mongo_client[ChatbotConfig.MONGO_DB_NAME]
        conversations_collection = mongo_db[ChatbotConfig.CONVERSATIONS_COLLECTION_NAME]
        
        # Có thể thêm các index nếu cần thiết
        # await conversations_collection.create_index("user_id")
        # await conversations_collection.create_index("timestamp")

        logging.info("MongoDB connection established.")
    except Exception as e:
        logging.critical(f"MongoDB connection failed: {str(e)}. Application may not function correctly.")
        mongo_client = None
        mongo_db = None
        conversations_collection = None
    # --- KẾT THÚC THAY THẾ ---

    # ... (phần Model loading hiện có) ...
    logging.info("Loading AI model...")
    # Bạn cần đảm bảo model_manager được khởi tạo trước khi gọi load_model
    # Nếu model_manager chưa được định nghĩa ở đây, hãy định nghĩa nó
    # (nó nên được định nghĩa ở bước 8, sau phần global vars)
    if not await model_manager.load_model():
        logging.critical("Failed to load AI model. Application will not function correctly.")

# 8. MODEL SETUP
class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path("training/results/best_model")
        self.model = None
        self.tokenizer = None
    
    async def load_model(self):
        try:
            # Kiểm tra xem thư mục model có tồn tại và có chứa các file cần thiết không
            if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
                logging.error(f"Model directory not found or is empty: {self.model_dir}")
                # Fallback hoặc yêu cầu tải model
                # Ví dụ: return await self._download_model_if_missing()
                return False

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                local_files_only=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_dir,
                local_files_only=True
            ).to(self.device)
            self.model.eval()
            logging.info(f"Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logging.error(f"Model loading error: {str(e)}")
            return False

model_manager = ModelManager()


# 9. UTILITY FUNCTIONS
def preprocess_question(text: str) -> str:
    """Tiền xử lý câu hỏi"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s?!,.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    if any(q in text for q in ['ai', 'làm sao', 'như thế nào', 'gì', 'bao giờ', 'ở đâu']):
        if not text.endswith('?'):
            text += '?'
    return text

def get_intent(text: str) -> str:
    """Phân tích ý định chi tiết"""
    text = text.lower()
    intents = {
        'greeting': ['xin chào', 'hello', 'hi', 'chào', 'alo'],
        'farewell': ['tạm biệt', 'bye', 'goodbye', 'gặp lại'],
        'question': ['là gì', 'như thế nào', 'bao giờ', 'thế nào', 'làm sao', 'tại sao'],
        'thanks': ['cảm ơn', 'thanks', 'thank', 'cảm tạ'],
        'agreement': ['đồng ý', 'được', 'ok', 'ừ', 'vâng'],
        'disagreement': ['không', 'chưa', 'không đồng ý'],
        'help': ['giúp', 'hỗ trợ', 'support'],
        'complain': ['khiếu nại', 'phàn nàn', 'không hài lòng']
    }
    
    for intent, patterns in intents.items():
        if any(p in text for p in patterns):
            return intent
    return 'other'

def enhance_context(history: List[Dict], current_input: str) -> Dict[str, Any]:
    """Tạo ngữ cảnh phong phú"""
    current_time = datetime.now()
    time_context = (
        "buổi sáng" if 5 <= current_time.hour < 12
        else "buổi chiều" if 12 <= current_time.hour < 18
        else "buổi tối"
    )
    
    conversation_state = "follow_up" if history else "new"
    
    topics = {
        'tech': ['máy tính', 'điện thoại', 'phần mềm', 'ứng dụng'],
        'general': ['thời tiết', 'tin tức', 'thể thao'],
        'personal': ['bạn', 'tôi', 'chúng ta']
    }
    
    current_topic = next(
        (topic for topic, keywords in topics.items() 
         if any(keyword in current_input.lower() for keyword in keywords)),
        'unknown'
    )
    
    context = {
        "time": time_context,
        "datetime": current_time.isoformat(),
        "intent": get_intent(current_input),
        "conversation_state": conversation_state,
        "topic": current_topic,
        "preprocessed_input": preprocess_question(current_input)
    }
    
    if history:
        last_exchange = history[-1]
        context.update({
            "last_question": last_exchange['user'],
            "last_response": last_exchange['bot'],
            "conversation_length": len(history)
        })
    
    return context

async def generate_response(
    text: str,
    context: Dict[str, Any],
    intent: str,
    temperature: float = ChatbotConfig.TEMPERATURE
) -> str:
    """Generate câu trả lời tự nhiên"""
    try:
        if not model_manager.model:
            if not await model_manager.load_model():
                return random.choice(FALLBACK_RESPONSES)
        
        prompt = f"""
        Context: {json.dumps(context, ensure_ascii=False)}
        Question: {text}
        Intent: {intent}
        
        Hãy trả lời một cách tự nhiên, thân thiện và chính xác.
        Answer: """
        
        inputs = model_manager.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=ChatbotConfig.MAX_LENGTH
        ).to(model_manager.device)

        with torch.no_grad():
            outputs = model_manager.model.generate(
                **inputs,
                max_length=ChatbotConfig.MAX_LENGTH,
                min_length=ChatbotConfig.MIN_LENGTH,
                num_beams=ChatbotConfig.NUM_BEAMS,
                no_repeat_ngram_size=ChatbotConfig.NO_REPEAT_NGRAM_SIZE,
                repetition_penalty=ChatbotConfig.REPETITION_PENALTY,
                length_penalty=ChatbotConfig.LENGTH_PENALTY,
                early_stopping=True,
                do_sample=True,
                temperature=temperature,
                top_k=ChatbotConfig.TOP_K,
                top_p=ChatbotConfig.TOP_P
            )

        response = model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace("Answer:", "").strip()
        
        if not response.endswith(('.', '!', '?')):
            response += '.'
            
        return response
        
    except Exception as e:
        logging.error(f"Generation error: {str(e)}")
        return random.choice(FALLBACK_RESPONSES)

# 10. API ENDPOINTS
# --- THAY ĐỔI ENDPOINT /CHAT ĐỂ SỬ DỤNG MONGODB ---
@app.post("/chat")
async def chat_response(
    chat_input: ChatInput,
    background_tasks: BackgroundTasks,
    # Áp dụng Rate Limiter nếu USE_REDIS = True
    dependencies=[Depends(RateLimiter(times=ChatbotConfig.RATE_LIMIT_TIMES, minutes=ChatbotConfig.RATE_LIMIT_MINUTES))] if USE_REDIS else []
):
    """Main chat endpoint"""
    try:
        user_text = preprocess_question(chat_input.user_input)
        
        # --- Redis Cache Check (Nếu bạn đã thêm vào từ gợi ý trước) ---
        cache_key = f"chat:{hash(user_text)}"
        if USE_REDIS and redis_client:
            cached_response = await redis_client.get(cache_key)
            if cached_response:
                logging.info(f"Response from cache for user_input: {user_text}")
                return JSONResponse(content=json.loads(cached_response))
        # -----------------------------------------------------------

        intent = get_intent(user_text)
        context = enhance_context(chat_input.conversation_history, user_text)
        
        # Handle basic intents
        if intent == 'greeting':
            time_context = context['time']
            responses = [
                f"Xin chào! Chúc bạn có một {time_context} tốt lành!",
                f"Chào bạn! Rất vui được gặp bạn trong {time_context} này.",
                f"Xin chào! Tôi có thể giúp gì cho bạn trong {time_context} này?"
            ]
            response = random.choice(responses)
        elif intent == 'farewell':
            response = random.choice([
                "Tạm biệt! Hẹn gặp lại bạn sau nhé!",
                "Chào tạm biệt! Rất vui được trò chuyện với bạn!",
                "Goodbye! Chúc bạn có một ngày tốt lành!"
            ])
        elif intent == 'thanks':
            response = random.choice([
                "Không có gì! Rất vui được giúp bạn.",
                "Không có chi! Đó là nhiệm vụ của tôi.",
                "Rất vui vì đã giúp được bạn!"
            ])
        else:
            response = await generate_response(user_text, context, intent)
        
        # --- Logic để đánh giá 'is_good_data' bằng heuristic (tùy chọn) ---
        # is_good_data = evaluate_response_quality(user_text, response, context, intent)
        # ------------------------------------------------------------------

        # --- LƯU HỘI THOẠI VÀO MONGODB ---
        conversation_id = None
        if conversations_collection: # Chỉ lưu nếu kết nối DB thành công
            try:
                # user_id có thể lấy từ token hoặc session nếu bạn có authentication
                conversation_record = {
                    "user_id": None, # Đặt user_id thực tế nếu có
                    "timestamp": datetime.now(timezone.utc), # Lưu UTC time
                    "user_input": user_text,
                    "bot_response": response,
                    "intent": intent,
                    "context": context,
                    "feedback_score": None,
                    "feedback_timestamp": None,
                    "is_good_data": False # Mặc định là False, sẽ được cập nhật bởi feedback
                    # "is_good_data": is_good_data # Nếu dùng heuristic
                }
                result = await conversations_collection.insert_one(conversation_record)
                conversation_id = str(result.inserted_id) # MongoDB ObjectId là string
                logging.info(f"Conversation logged with ID: {conversation_id}")
            except Exception as db_e:
                logging.error(f"Failed to log conversation to MongoDB: {str(db_e)}")
        # --- KẾT THÚC LƯU VÀO MONGODB ---
        
        # Update history
        updated_history = chat_input.conversation_history + [
            {"user": user_text, "bot": response}
        ][-ChatbotConfig.MAX_HISTORY:]
        
        result = {
            "response": response,
            "conversation_history": updated_history,
            "intent": intent,
            "context": context,
            "conversation_id": conversation_id # Trả về ID để người dùng có thể gửi feedback
        }
        
        # --- Redis Cache Set (Nếu bạn đã thêm vào từ gợi ý trước) ---
        if USE_REDIS and redis_client:
            try:
                background_tasks.add_task(
                    redis_client.set,
                    cache_key,
                    json.dumps(result, ensure_ascii=False, default=str), # Thêm default=str để xử lý ObjectId nếu có
                    ex=ChatbotConfig.CACHE_TTL
                )
            except Exception as e:
                logging.error(f"Caching error: {str(e)}")
        # -----------------------------------------------------------
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        return JSONResponse(
            content={
                "response": random.choice(FALLBACK_RESPONSES),
                "conversation_history": chat_input.conversation_history,
                "intent": "error",
                "conversation_id": None # Không có ID nếu lỗi
            },
            status_code=500
        )

# --- THAY ĐỔI ENDPOINT /FEEDBACK ĐỂ SỬ DỤNG MONGODB ---
@app.post("/feedback")
async def submit_feedback(feedback_input: FeedbackInput):
    """Endpoint để người dùng gửi feedback về một phản hồi của bot."""
    if not conversations_collection:
        raise HTTPException(status_code=500, detail="MongoDB collection not available for feedback.")

    try:
        # MongoDB sử dụng ObjectId cho _id, cần chuyển đổi string sang ObjectId
        conversation_obj_id = ObjectId(feedback_input.conversation_id)

        # Cập nhật bản ghi trong collection
        update_result = await conversations_collection.update_one(
            {"_id": conversation_obj_id},
            {
                "$set": {
                    "feedback_score": feedback_input.score,
                    "feedback_timestamp": datetime.now(timezone.utc), # Lưu UTC time
                    "is_good_data": feedback_input.score >= 1 # Đánh dấu là dữ liệu tốt nếu điểm >= 1
                }
            }
        )

        if update_result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        logging.info(f"Feedback received for conversation ID {feedback_input.conversation_id}: Score={feedback_input.score}")
        return {"message": "Feedback submitted successfully", "conversation_id": feedback_input.conversation_id}
    except HTTPException as http_e:
        raise http_e # Re-raise HTTPException
    except Exception as e:
        logging.error(f"Error submitting feedback for conversation ID {feedback_input.conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")
# --- KẾT THÚC THAY ĐỔI ---

# ... (các endpoint khác như / và /health) ...
# Thêm route mới cho trang chủ
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve trang chủ"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# Thêm route xử lý lỗi
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Không tìm thấy endpoint này",
            "detail": "Vui lòng kiểm tra lại URL"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        memory_info = {
            "allocated": torch.cuda.memory_allocated(),
            "cached": torch.cuda.memory_reserved()
        } if torch.cuda.is_available() else None
        
        # --- THAY ĐỔI CHECK MONGODB HEALTH ---
        mongo_status = False
        if mongo_client:
            try:
                # Lệnh ping đơn giản để kiểm tra kết nối MongoDB
                await mongo_client.admin.command('ping')
                mongo_status = True
            except Exception:
                mongo_status = False
        # --- KẾT THÚC THAY ĐỔI ---

        return {
            "status": "healthy",
            "model": {
                "name": str(model_manager.model_dir),
                "device": str(model_manager.device),
                "loaded": model_manager.model is not None
            },
            "memory": memory_info,
            "redis": await redis_client.ping() if USE_REDIS and redis_client else False, # Kiểm tra redis chỉ khi USE_REDIS là True
            "mongodb": mongo_status, # Thêm trạng thái MongoDB
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

# 11. SHUTDOWN EVENT
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup khi shutdown"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if USE_REDIS and redis_client: # Đóng redis chỉ khi USE_REDIS là True
        await redis_client.close()
    
    # --- THAY ĐỔI PHẦN SHUTDOWN MONGODB ---
    if mongo_client:
        logging.info("Closing MongoDB connection.")
        mongo_client.close()
    # --- KẾT THÚC THAY ĐỔI ---

# 12. MAIN
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
