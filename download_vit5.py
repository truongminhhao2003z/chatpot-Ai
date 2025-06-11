import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Tên mô hình bạn muốn tải
model_name = "vietai/vit5-base"

# Đường dẫn đến thư mục cache
# Thư mục này sẽ được tạo nếu chưa tồn tại
cache_dir = "./model_cache" # Hoặc bạn có thể chọn đường dẫn tuyệt đối, ví dụ: "/path/to/your/model_cache"

# Tạo thư mục cache nếu nó chưa tồn tại
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Đã tạo thư mục: {cache_dir}")

# Tải tokenizer và lưu vào cache_dir
print(f"Đang tải tokenizer cho mô hình {model_name} và lưu vào '{cache_dir}'...")
# Thanh tiến trình sẽ tự động hiển thị khi tải
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print("Tải tokenizer hoàn tất.")

# Tải mô hình và lưu vào cache_dir
print(f"Đang tải mô hình {model_name} và lưu vào '{cache_dir}'...")
# Thanh tiến trình sẽ tự động hiển thị khi tải
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
print("Tải mô hình hoàn tất.")

print(f"\nBạn đã tải thành công tokenizer và mô hình vietai/vit5-base vào thư mục '{cache_dir}'.")