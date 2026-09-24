import gzip
import json
from pathlib import Path

file_path = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "c4-train.00000-of-01024-30K.json.gz"
)

documents = []
with gzip.open(file_path, "rt", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        # Giả định mỗi line có trường 'text'
        if "text" in data:
            documents.append(data["text"])

print(f"Tổng số tài liệu đã load: {len(documents)}")
# Xem thử tài liệu đầu tiên
print("Mẫu tài liệu 1:", documents[0][:200], "...")