# 1. IMPORTS
from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)
import os, uvicorn, logging, re, random, asyncio, json, uuid
from datetime import datetime, timezone
import redis.asyncio as redis
from typing import List, Dict, Optional, Union, Any, Callable, AsyncGenerator
from pathlib import Path
import numpy as np
from functools import lru_cache
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from bson import ObjectId
from pydantic_core import core_schema, PydanticCustomError
from contextlib import asynccontextmanager
import traceback

# 2. CONFIG CLASS
USE_REDIS = True
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

    # MongoDB config
    MONGO_DB_URL = "mongodb://localhost:27017/"
    MONGO_DB_NAME = "chatbot_db"
    CONVERSATIONS_COLLECTION = "conversations"
    MESSAGES_COLLECTION = "chat_messages"
    SESSIONS_COLLECTION = "chat_sessions"
    MONGO_TIMEOUT_MS = 5000

# 3. LIFESPAN EVENT HANDLER
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client, mongo_client
    
    # Initialize Redis
    if USE_REDIS:
        try:
            redis_client = redis.from_url(
                ChatbotConfig.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=ChatbotConfig.REDIS_TIMEOUT
            )
            await FastAPILimiter.init(redis_client)
            logging.info("Redis connection established")
        except Exception as e:
            logging.error(f"Redis connection failed: {str(e)}")
            redis_client = None

    # Initialize MongoDB
    try:
        mongo_client = AsyncIOMotorClient(
            ChatbotConfig.MONGO_DB_URL,
            serverSelectionTimeoutMS=ChatbotConfig.MONGO_TIMEOUT_MS
        )
        await mongo_client.admin.command('ping')
        
        # Initialize all MongoDB managers
        await conversation_manager.connect_db(mongo_client)
        await message_manager.connect_db(mongo_client)
        await session_manager.connect_db(mongo_client)
            
        logging.info("MongoDB connection established")
    except Exception as e:
        logging.critical(f"MongoDB connection failed: {str(e)}")
        mongo_client = None

    # Load AI model
    if not await model_manager.load_model():
        logging.critical("Failed to load AI model")
    
    yield
    
    # Shutdown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if USE_REDIS and redis_client:
        await redis_client.close()
    if mongo_client:
        mongo_client.close()
        logging.info("MongoDB connection closed")

# 4. INITIALIZATION
app = FastAPI(
    title="Chatbot API with MongoDB",
    description="API for AI Chatbot using transformer and MongoDB",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 5. PYDANTIC MODELS
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

        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.no_info_plain_validator_function(validate_from_anything)
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda o: str(o))
        )

class ChatInput(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = Field(None, description="ID phiên chat")
    user_id: Optional[str] = Field(None, description="ID người dùng")
    language: str = Field(default="vi")

    @field_validator('user_input')
    @classmethod
    def validate_input(cls, v):
        if not v.strip():
            raise ValueError('Input cannot be empty')
        return v.strip()

class ChatMessage(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    session_id: str
    user_id: Optional[str]
    message_type: str  # "user" or "bot"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class ChatSession(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    session_id: str
    user_id: Optional[str]
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0
    is_active: bool = True

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# 6. DATABASE MANAGERS
class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path("training/results/best_model")
        self.model = None
        self.tokenizer = None
    
    async def load_model(self) -> bool:
        try:
            if not self.model_dir.exists():
                logging.warning(f"Model directory not found: {self.model_dir}")
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

class ConversationManager:
    def __init__(self):
        self.collection: Optional[AsyncIOMotorCollection] = None
    
    async def connect_db(self, client: AsyncIOMotorClient):
        try:
            self.collection = client[ChatbotConfig.MONGO_DB_NAME][ChatbotConfig.CONVERSATIONS_COLLECTION]
            await self.collection.create_index("session_id")
            await self.collection.create_index("user_id")
            await self.collection.create_index("timestamp")
            logging.info("Conversation manager connected to MongoDB")
            return True
        except Exception as e:
            logging.error(f"Conversation manager connection error: {str(e)}")
            return False

conversation_manager = ConversationManager()

class MessageManager:
    def __init__(self):
        self.collection: Optional[AsyncIOMotorCollection] = None
    
    async def connect_db(self, client: AsyncIOMotorClient):
        try:
            self.collection = client[ChatbotConfig.MONGO_DB_NAME][ChatbotConfig.MESSAGES_COLLECTION]
            await self.collection.create_index("session_id")
            await self.collection.create_index("user_id")
            await self.collection.create_index("timestamp")
            await self.collection.create_index([("session_id", 1), ("timestamp", 1)])
            logging.info("Message manager connected to MongoDB")
            return True
        except Exception as e:
            logging.error(f"Message manager connection error: {str(e)}")
            return False
    
    async def save_message(self, message_data: dict) -> str:
        try:
            result = await self.collection.insert_one(message_data)
            return str(result.inserted_id)
        except Exception as e:
            logging.error(f"Error saving message: {str(e)}")
            raise
    
    async def get_session_messages(self, session_id: str, limit: int = 100) -> List[Dict]:
        try:
            cursor = self.collection.find(
                {"session_id": session_id}
            ).sort("timestamp", 1).limit(limit)
            
            return [doc async for doc in cursor]
        except Exception as e:
            logging.error(f"Error fetching messages: {str(e)}")
            return []

message_manager = MessageManager()

class SessionManager:
    def __init__(self):
        self.collection: Optional[AsyncIOMotorCollection] = None
    
    async def connect_db(self, client: AsyncIOMotorClient):
        try:
            self.collection = client[ChatbotConfig.MONGO_DB_NAME][ChatbotConfig.SESSIONS_COLLECTION]
            await self.collection.create_index("session_id", unique=True)
            await self.collection.create_index("user_id")
            await self.collection.create_index("last_activity")
            logging.info("Session manager connected to MongoDB")
            return True
        except Exception as e:
            logging.error(f"Session manager connection error: {str(e)}")
            return False
    
    async def create_session(self, user_id: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "start_time": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "message_count": 0,
            "is_active": True
        }
        await self.collection.insert_one(session_data)
        return session_id
    
    async def update_session(self, session_id: str):
        await self.collection.update_one(
            {"session_id": session_id},
            {
                "$set": {"last_activity": datetime.now(timezone.utc)},
                "$inc": {"message_count": 1}
            }
        )
    
    async def close_session(self, session_id: str):
        await self.collection.update_one(
            {"session_id": session_id},
            {"$set": {"is_active": False}}
        )
    
    async def get_session(self, session_id: str) -> Optional[Dict]:
        return await self.collection.find_one({"session_id": session_id})

session_manager = SessionManager()

# 7. UTILITY FUNCTIONS
def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'chatbot_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )

setup_logging()

FALLBACK_RESPONSES = [
    "Sorry, I'm having trouble. Please try again later.",
    "I can't respond right now. Please bear with me.",
    "Something went wrong. Try asking me something else!",
    "I'm still learning. Please try again in a few minutes."
]

def preprocess_question(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s?!,.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    if any(q in text for q in ['who', 'how', 'what', 'when', 'where']):
        if not text.endswith('?'):
            text += '?'
    return text

def get_intent(text: str) -> str:
    text = text.lower()
    intents = {
        'greeting': ['hello', 'hi', 'hey', 'greetings'],
        'farewell': ['goodbye', 'bye', 'see you', 'farewell'],
        'question': ['what', 'how', 'when', 'where', 'why', 'who'],
        'thanks': ['thank you', 'thanks', 'appreciate'],
        'help': ['help', 'support', 'assist']
    }
    
    for intent, patterns in intents.items():
        if any(p in text for p in patterns):
            return intent
    return 'other'

def enhance_context(history: List[Dict], current_input: str) -> Dict[str, Any]:
    current_time = datetime.now()
    time_context = (
        "morning" if 5 <= current_time.hour < 12
        else "afternoon" if 12 <= current_time.hour < 18
        else "evening"
    )
    
    context = {
        "time": time_context,
        "datetime": current_time.isoformat(),
        "intent": get_intent(current_input),
        "conversation_state": "follow_up" if history else "new",
        "preprocessed_input": preprocess_question(current_input)
    }
    
    if history:
        context.update({
            "last_question": history[-1].get("content", "") if history[-1].get("message_type") == "user" else "",
            "last_response": history[-1].get("content", "") if history[-1].get("message_type") == "bot" else "",
            "conversation_length": len(history)
        })
    
    return context

async def generate_response(text: str, context: Dict[str, Any], intent: str) -> str:
    try:
        if not model_manager.model:
            if not await model_manager.load_model():
                return random.choice(FALLBACK_RESPONSES)
        
        prompt = f"""
        Context: {json.dumps(context, ensure_ascii=False)}
        Question: {text}
        Intent: {intent}
        
        Please respond naturally and helpfully.
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
                temperature=ChatbotConfig.TEMPERATURE,
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

# 8. API ENDPOINTS
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_response(
    chat_input: ChatInput,
    background_tasks: BackgroundTasks,
    request: Request
):
    try:
        logging.info(f"📥 Nhận input từ client: {chat_input}")
        
        # Step 1: Validate and get session
        session_id = chat_input.session_id or await session_manager.create_session(chat_input.user_id)
        logging.info(f"📌 Session ID: {session_id}")
        
        session = await session_manager.get_session(session_id)
        if not session:
            session_id = await session_manager.create_session(chat_input.user_id)
            session = await session_manager.get_session(session_id)
            logging.info(f"📌 Tạo session mới: {session_id}")
        
        # Step 2: Get previous messages for context
        previous_messages = await message_manager.get_session_messages(session_id)
        logging.info(f"📌 Lấy {len(previous_messages)} tin nhắn cũ cho context.")

        # Step 3: Preprocess input
        user_text = preprocess_question(chat_input.user_input)
        logging.info(f"📌 Câu hỏi người dùng: {user_text}")

        
        # Step 4: Save user message
        user_message_data = {
            "session_id": session_id,
            "user_id": chat_input.user_id,
            "message_type": "user",
            "content": user_text,
            "metadata": {
                "ip_address": request.client.host,
                "user_agent": request.headers.get("user-agent", ""),
                "language": chat_input.language
            }
        }
        await message_manager.save_message(user_message_data)
        
        # Step 5: Generate context and response
        context = enhance_context(previous_messages, user_text)
        intent = get_intent(user_text)
        
        if intent == 'greeting':
            response = f"Hello! Good {context['time']}!"
        elif intent == 'farewell':
            response = "Goodbye! Have a nice day!"
            await session_manager.close_session(session_id)
        elif intent == 'thanks':
            response = "You're welcome!"
        else:
            response = await generate_response(user_text, context, intent)
        
        # Step 6: Save bot response
        bot_message_data = {
            "session_id": session_id,
            "user_id": chat_input.user_id,
            "message_type": "bot",
            "content": response,
            "metadata": {
                "intent": intent,
                "context": context
            }
        }
        await message_manager.save_message(bot_message_data)
        
        # Step 7: Update session activity
        await session_manager.update_session(session_id)
        
        # Step 8: Get updated message history
        updated_messages = await message_manager.get_session_messages(session_id)

        # Chuyển tất cả ObjectId thành chuỗi
        for msg in updated_messages:
            if "_id" in msg:
                msg["_id"] = str(msg["_id"])

        
        # Prepare response
        return JSONResponse(content={
            "session_id": session_id,
            "response": response,
            "messages": updated_messages[-ChatbotConfig.MAX_HISTORY:],
            "intent": intent,
            "context": context
        })
        
    except Exception as e:
        logging.error(f"Chat error: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            content={
                "error": "Internal server error",
                "message": str(e),
                "fallback_response": random.choice(FALLBACK_RESPONSES)
            },
            status_code=500
        )

@app.get("/history/{session_id}", response_model=List[ChatMessage])
async def get_chat_history(session_id: str, limit: int = 100):
    try:
        messages = await message_manager.get_session_messages(session_id, limit)
        return messages
    except Exception as e:
        logging.error(f"History error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching chat history")

@app.post("/sessions/{session_id}/close")
async def close_chat_session(session_id: str):
    try:
        await session_manager.close_session(session_id)
        return {"status": "success", "session_id": session_id}
    except Exception as e:
        logging.error(f"Session close error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error closing session")

@app.get("/health")
async def health_check():
    try:
        services = {
            "mongodb": False,
            "redis": False,
            "model": model_manager.model is not None
        }
        
        if mongo_client:
            try:
                await mongo_client.admin.command('ping')
                services["mongodb"] = True
            except Exception:
                pass
        
        if USE_REDIS and redis_client:
            try:
                await redis_client.ping()
                services["redis"] = True
            except Exception:
                pass
        
        status = "healthy" if all(services.values()) else "degraded"
        
        return {
            "status": status,
            "services": services,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

# 9. MAIN

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
