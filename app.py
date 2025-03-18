from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Khởi tạo FastAPI
app = FastAPI()

# Load mô hình chatbot
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Kết nối thư mục templates và static
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Định nghĩa dữ liệu đầu vào
class ChatInput(BaseModel):
    user_input: str

# API chatbot xử lý tin nhắn từ người dùng
@app.post("/chat")
def chat_response(chat_input: ChatInput):
    inputs = tokenizer(chat_input.user_input, return_tensors="pt", max_length=512, truncation=True)
    reply_ids = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return {"response": response}

# Endpoint phục vụ trang web
@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Chạy server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
