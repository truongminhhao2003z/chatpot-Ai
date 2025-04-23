# 1. IMPORTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderCache, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import json, re, os, warnings, logging
from datetime import datetime

# 2. LOGGING CONFIGURATION 
log_dir = "training/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# 3. PREPROCESSING FUNCTIONS
def preprocess_input(text):
    """Xử lý text đầu vào: lowercase và loại bỏ ký tự đặc biệt"""
    text = text.strip().lower()
    text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', '', text)
    return text

# 4. DATASET CLASS
class ChatbotDataset(Dataset):
    """Custom Dataset cho chatbot"""
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

        # Tokenize input và target
        input_encodings = self.tokenizer(
            input_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        target_encodings = self.tokenizer(
            target_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        # Xử lý encodings
        input_ids = input_encodings['input_ids'].squeeze()
        attention_mask = input_encodings['attention_mask'].squeeze()
        labels = target_encodings['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# 5. MODEL EVALUATION
def evaluate_model(model, dataloader, device):
    """Đánh giá model trên validation set"""
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
    # 6.1 Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Đang sử dụng: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 2
    MAX_LENGTH = 64
    NUM_EPOCHS = 5
    
    # Directories
    cache_dir = "model_cache"
    results_dir = "training/results"
    os.makedirs(results_dir, exist_ok=True)

    # 6.2 Model Loading
    print("\n1️⃣ Đang tải model từ cache...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(f"{cache_dir}/tokenizer", local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            f"{cache_dir}/model",
            local_files_only=True,
            use_cache=False
        ).to(device)
        print("✅ Tải model thành công!")
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        return

    # 6.3 Data Loading
    print("\n2️⃣ Đang tải dữ liệu...")
    try:
        with open('training/data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Đã tải {len(data)} mẫu dữ liệu")
    except FileNotFoundError:
        print("❌ Không tìm thấy file data.json!")
        return

    # 6.4 Data Preparation
    print("\n3️⃣ Chuẩn bị dữ liệu...")
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

    # 6.5 Optimizer & Scheduler Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * NUM_EPOCHS
    )

    # 6.6 Training Loop
    print(f"\n4️⃣ Bắt đầu training...")
    print(f"📊 Train: {len(train_dataset)} mẫu, Validation: {len(val_dataset)} mẫu")
    print(f"📈 Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")

    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        
        for batch in progress_bar:
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            
            # Backward pass
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            
            # Optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        val_loss = evaluate_model(model, val_dataloader, device)
        print(f"\n📊 Epoch {epoch+1}: Train Loss = {total_loss/len(train_dataloader):.4f}, Val Loss = {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("💾 Lưu model tốt nhất...")
            model.save_pretrained(f"{results_dir}/best_model")
            tokenizer.save_pretrained(f"{results_dir}/best_model")

    # 6.7 Model Testing
    print("\n5️⃣ Test model...")
    test_inputs = [
        "Xin chào!",
        "2 + 2 bằng mấy?",
        "Hôm nay thời tiết thế nào?"
    ]

    model.eval()
    for test_input in test_inputs:
        print(f"\nInput: {test_input}")
        processed_input = preprocess_input(test_input)
        inputs = tokenizer(
            f"chat: {processed_input}", 
            return_tensors="pt", 
            max_length=64, 
            padding=True, 
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=64,
                num_beams=3,
                early_stopping=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: {response}")

    print("\n✨ Training hoàn tất!")

# 7. ENTRY POINT
if __name__ == "__main__":
    main()
