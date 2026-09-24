import numpy as np
from sentence_transformers import SentenceTransformer
from data_loader import documents

# Load pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Để tăng tốc độ xử lý 30.000 docs, nên dùng GPU nếu có (device='cuda')
# Thực hiện encoding theo batch
batch_size = 128
embeddings = model.encode(
    documents, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
)

print(f"Kích thước ma trận Embeddings: {embeddings.shape}")