# LAB 02 — Language Models

## Cấu trúc thư mục

```text
lab02/
├── README.md
├── c4-train.00000-of-01024-30K.json.gz
├── calculations.pdf
├── prediction.pdf
├── ngram_lm.py
├── experiments.ipynb
└── results.csv
```

## Dữ liệu

File:

```text
c4-train.00000-of-01024-30K.json.gz
```

## Các phần đã chuẩn bị

### Section 13 — Corpus statistics

Notebook tính:

- số documents
- số sentences
- số tokens
- vocabulary size
- unique unigram/bigram/trigram
- singleton unigram/bigram/trigram
- frequency distributions

### Sections 14–15 — Core implementation + log probability

`ngram_lm.py` triển khai từ đầu:

- `build_vocabulary()`
- `count_ngrams()`
- `train/fit` cho unigram, bigram, trigram
- `probability()`
- `sentence_probability()`
- `sentence_log_probability()`
- `perplexity()`
- `next_word_distribution()`
- `rank_candidates()`

Có cả MLE và Laplace smoothing.

### Section 16 — MLE vs Laplace

Notebook tạo bảng MLE/Laplace trên:

- train
- validation
- test

cho unigram/bigram/trigram.

### Section 17 & 19 — Perplexity

Notebook tính perplexity thống nhất cách tokenization, boundary và EOS giữa các mô hình.

### Sections 20–21 — Prediction + ranking

Notebook có sẵn cell để:

- chọn ít nhất 5 contexts
- dự đoán next word
- ghi actual next word
- xếp hạng candidate continuations

## Chạy bài

Từ thư mục `lab02`:

```bash
jupyter notebook experiments.ipynb
```

hoặc:

```bash
jupyter lab
```

Sau đó chạy các cell theo thứ tự.

Notebook sẽ xuất/ghi lại kết quả vào:

```text
results.csv
```