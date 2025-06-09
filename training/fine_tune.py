# 1. IMPORTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import json, re, os, logging
from datetime import datetime

# 2. LOGGING CONFIGURATION
log_dir = "training/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 3. PREPROCESSING FUNCTIONS
def preprocess_input(text):
    """
    Xử lý text đầu vào:
    - chuyển thành chữ thường
    - loại bỏ ký tự đặc biệt không mong muốn (ngoại trừ tiếng Việt)
    """
    text = text.strip().lower()
    # Giữ lại chữ cái, số, dấu cách và các ký tự tiếng Việt cơ bản
    text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', '', text)
    return text

# 4. DATASET CLASS
class ChatbotDataset(Dataset):
    """Dataset tùy chỉnh cho chatbot"""

    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"chat: {item['input']}"
        target_text = item['output']

        # Tokenize input
        input_encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Tokenize target (label)
        target_encodings = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = input_encodings['input_ids'].squeeze()
        attention_mask = input_encodings['attention_mask'].squeeze()
        labels = target_encodings['input_ids'].squeeze()

        # Bỏ qua padding token trong loss calculation bằng cách thay thế bằng -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# 5. MODEL EVALUATION
def evaluate_model(model, dataloader, device):
    """Đánh giá model trên tập validation"""
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()

    return total_val_loss / len(dataloader)

# 6. MAIN TRAINING FUNCTION
def main():
    # 6.1 Setup thiết bị và siêu tham số
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Đang sử dụng: {device}")
    logging.info(f"Đang sử dụng thiết bị: {device}")

    BATCH_SIZE = 2
    MAX_LENGTH = 64
    NUM_EPOCHS = 3

    cache_dir = "./model_cache"
    results_dir = "training/results"
    os.makedirs(results_dir, exist_ok=True)

    model_name_or_path = "vietai/vit5-base"

    # 6.2 Load model và tokenizer
    print(f"\n1️⃣ Đang tải model '{model_name_or_path}'...")
    logging.info(f"Đang tải model '{model_name_or_path}' từ Hugging Face Hub hoặc cache tại '{cache_dir}'...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=cache_dir).to(device)
        print("✅ Tải model thành công!")
        logging.info("Tải model thành công.")
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        logging.error(f"Lỗi khi tải model: {str(e)}")
        print("Vui lòng kiểm tra kết nối internet hoặc đảm bảo mô hình đã được tải về đầy đủ trong cache.")
        return

    # 6.3 Load dữ liệu
    print("\n2️⃣ Đang tải dữ liệu...")
    logging.info("Đang tải dữ liệu từ 'training/data.json'...")

    try:
        with open('training/data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Đã tải {len(data)} mẫu dữ liệu")
        logging.info(f"Đã tải {len(data)} mẫu dữ liệu.")
    except FileNotFoundError:
        print("❌ Không tìm thấy file data.json! Vui lòng đảm bảo file tồn tại.")
        logging.error("Không tìm thấy file data.json!")
        return

    # 6.4 Chuẩn bị dataset và dataloader
    print("\n3️⃣ Chuẩn bị dữ liệu...")
    logging.info("Đang chuẩn bị dữ liệu cho training và validation.")

    dataset = ChatbotDataset(data, tokenizer, max_length=MAX_LENGTH)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True
    )
    logging.info(f"Phân chia dữ liệu: Train {len(train_dataset)} mẫu, Validation {len(val_dataset)} mẫu.")

    # 6.5 Optimizer và Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * NUM_EPOCHS
    )
    logging.info("Đã thiết lập Optimizer và Scheduler.")

    # 6.6 Vòng lặp huấn luyện
    print(f"\n4️⃣ Bắt đầu training...")
    print(f"📊 Train: {len(train_dataset)} mẫu, Validation: {len(val_dataset)} mẫu")
    print(f"📈 Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
    logging.info(f"Bắt đầu training với {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}.")

    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')

        for batch in progress_bar:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )

            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

        val_loss = evaluate_model(model, val_dataloader, device)
        print(f"\n📊 Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        logging.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")

        # Lưu model tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("💾 Lưu model tốt nhất...")
            logging.info(f"Val loss cải thiện ({best_val_loss:.4f}), lưu model tốt nhất.")
            model.save_pretrained(f"{results_dir}/best_model")
            tokenizer.save_pretrained(f"{results_dir}/best_model")
        else:
            logging.info(f"Val loss không cải thiện. Best Val Loss hiện tại: {best_val_loss:.4f}.")

    # 6.7 Test model
    print("\n5️⃣ Test model...")
    logging.info("Bắt đầu test model với các ví dụ.")

    test_inputs = [
        "Xin chào!",
        "bye bye!",
        "Hôm nay thời tiết thế nào?",
        "Tôi muốn đăng ký nghỉ dài hạn cho học sinh"
    ]

    model.eval()
    for test_input in test_inputs:
        print(f"\nInput: {test_input}")
        processed_input = preprocess_input(test_input)
        inputs = tokenizer(
            f"chat: {processed_input}",
            return_tensors="pt",
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_beams=3,
                early_stopping=True
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: {response}")
        logging.info(f"Test Input: {test_input} -> Output: {response}")

    print("\n✨ Training hoàn tất!")
    logging.info("Training hoàn tất.")

# 7. ENTRY POINT
if __name__ == "__main__":
    main()
