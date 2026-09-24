import gzip
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

DATA_FILE = Path(__file__).parent / "data" / "c4-train.00000-of-01024-30K.json.gz"


def load_documents(file_path: Path) -> list[str]:
    documents = []
    with gzip.open(file_path, "rt", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            if "text" in data:
                documents.append(data["text"])
    return documents


def main() -> None:
    documents = load_documents(DATA_FILE)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        documents,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"Tổng số tài liệu đã load: {len(documents)}")
    print(f"Kích thước ma trận Embeddings: {embeddings.shape}")


if __name__ == "__main__":
    main()
