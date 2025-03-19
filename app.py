from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Khởi tạo FastAPI
app = FastAPI()

# Đường dẫn đến mô hình đã fine-tune
MODEL_PATH = "training/results/"

# Kiểm tra xem thư mục model có tồn tại không
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model directory '{MODEL_PATH}' not found!")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model: {str(e)}")

# Kết nối thư mục templates và static
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Định nghĩa dữ liệu đầu vào
class ChatInput(BaseModel):
    user_input: str

# API chatbot xử lý tin nhắn từ người dùng
@app.post("/chat")
def chat_response(chat_input: ChatInput):
    user_text = chat_input.user_input.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="❌ User input cannot be empty!")
    
    # Token hóa đầu vào
    inputs = tokenizer(user_text, return_tensors="pt", max_length=256, truncation=True)

    try:
        with torch.no_grad():  # Tắt gradient để tiết kiệm bộ nhớ
            reply_ids = model.generate(**inputs, max_length=100)
        response = tokenizer.decode(reply_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # Sửa lỗi mã hóa UTF-8
        response = response.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error generating response: {str(e)}")

# Endpoint phục vụ trang web
@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Chạy server FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
