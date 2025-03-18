from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# 1️⃣ Chọn mô hình (Phi-2)
model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2️⃣ Load dữ liệu huấn luyện
dataset = load_dataset("json", data_files="training/data.json")

# 3️⃣ Cấu hình LoRA (Fine-tune nhanh hơn)
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

# 4️⃣ Cấu hình huấn luyện
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# 5️⃣ Bắt đầu huấn luyện
trainer.train()
# 6️⃣ Lưu mô hình sau khi huấn luyện
trainer.save_model("training/results/")
