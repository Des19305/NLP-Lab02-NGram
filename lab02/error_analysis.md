# Error Analysis

---

## 1. Dự đoán đúng – Trường hợp 1

### the United → states

- Context: the United
- Prediction: states
- Actual: states
- Probability: 0.005183

### Phân tích
- Do context 'the United' cung cấp thông tin khá cụ thể cho từ tiếp theo nên khi mô hình sử dụng context này thì 'states' có xác suất cao nhất trong các candidate có thể xét. Vì vậy nó chọn 'states' và dự đoán đúng

---

## 2. Dự đoán đúng – Trường hợp 2

### it is → a

- Context: it is
- Prediction: a
- Actual: a
- Probability: 0.005124

### Phân tích
- Mô hình đã học được mối quan hệ giữa context 'it is' và từ tiếp theo 'a' từ training corpus. Khi tính xác suất cho các từ có thể xuất hiện sau context này, 'a' được mô hình lựa chọn và dự đoán chính xác. Tuy nhiên ta chưa thể kết luận 'it is a' là 1 trigram xuất hiện rất nhiều nếu chưa kiểm tra frequency/count cụ thể

---

## 3. Dự đoán sai - Trường hợp 1

### in the → world

- Context: in the
- Prediction: world
- Prediction probability: 0.003631
- Actual: los
- Actual probability: 0.000046

### Phân tích
- Vì mô hình thấy rằng xác suất của 'world' cao hơn nhiều so với của 'los' (0.003631 so với 0.000046, tức lớn hơn gấp khoảng 79 lần) vậy nên nó chọn 'world' và dự đoán sai

---

## 4.Dự đoán sai - Trường hợp 2

### this is → a

- Context: this is
- Model prediction: a
- Prediction probability: 0.006441
- Actual next word: already
- Actual probability: 0.000037

### Phân tích
- Tương tự như trường hợp 1, mô hình chọn 'a' và dự đoán sai vì nó nhận thấy xác suất của 'a' cao hơn đáng kể so với 'already' (lớn hơn gấp khoảng 174 lần)