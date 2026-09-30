# Reflection

---

## Câu 1
- Khi tăng n thì mô hình có được nhiều từ để sử dụng hơn trước đó để dự đoán từ tiếp theo


## Câu 2
- Khi n tăng, một n-gram phải có nhiều từ xuất hiện liên tiếp cùng nhau, từ đó làm sparsity tăng


## Câu 3
- Smoothing được dùng để giải quyết zero probability, không có smoothing thì các n-gram bị ẩn (không xuất hiện trong training set) sẽ rơi vào zero probability, qua đó làm cho xác suất của cả câu bằng 0 và perplexity trở thành vô cùng


## Câu 4
- Perplexity đo mức độ “bất ngờ” của model đối với dữ liệu
- Nếu model gán xác suất cao cho những từ thực tế xuất hiện
→ model ít bất ngờ
→ PPL thấp.

- Nếu model gán xác suất thấp
→ model rất bất ngờ
→ PPL cao.

## Câu 5
- Không, một model có PPL thấp có nghĩa là nó dự đoán tốt hơn theo metric xác suất trên dataset/evaluation setup đó. Nhưng con người còn quan tâm coherence (tính mạch lạc), meaning (ý nghĩa), grammar (ngữ pháp), relevance (sự liên quan), factuality (tính đúng đắn)

## Câu 6
- N-gram chỉ nhìn một số lượng từ cố định, do đó nó không hiểu toàn bộ câu
- N-gram không hiểu semantic meaning. Con người hiểu 1 từ có nghĩa khác nhau dựa trên ngữ cảnh
- Chuỗi chưa xuất hiện trong training sẽ làm cho n-gram khó xử lí
- N-gram cũng sẽ khó xử lí khi gặp long-range context

## Câu 7
- Không, mô hình trigram chỉ có thể dùng 2 từ trước đó để dự đoán từ tiếp theo, do đó nếu context gồm 100 từ thì 98 từ đầu tiên sẽ bị bỏ qua