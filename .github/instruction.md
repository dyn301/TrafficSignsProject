# Kế hoạch triển khai project: Tìm kiếm và nhận diện biển báo giao thông Việt Nam bằng Embeddings (CLIP + Vector Search)

## 1. Mục tiêu hệ thống
- Input: ảnh biển báo giao thông từ thư mục `./dataset_aug`
- Trích xuất embedding bằng mô hình CLIP (pretrained)
- Xây dựng vector database (FAISS/Milvus)
- Thực hiện truy vấn bằng cosine similarity
- Output:
  + Tên biển báo
  + Nhóm biển (cấm / nguy hiểm / hiệu lệnh / chỉ dẫn)
  + Ý nghĩa biển báo

## 2. Dữ liệu đầu vào
- Dataset ~3000 ảnh, ~300 loại biển
- Ảnh đã augment, phân loại theo thư mục từng loại biển
- Cấu trúc ban đầu:
  - `dataset_aug/Warning Signs/` ...
  - `dataset_aug/Mandatory Signs/` ...
  - `dataset_aug/Prohibitory Signs/` ...
  - `dataset_aug/Information Signs/` ...

## 3. Kiến trúc file notebook đề xuất
Danh sách notebook chính (mỗi file có nhiệm vụ rõ ràng):

1. `01_prepare_dataset.ipynb`
2. `02_extract_embedding.ipynb`
3. `03_build_vector_db.ipynb`
4. `04_search_demo.ipynb`
5. `05_evaluation.ipynb`
6. `06_api_demo.ipynb` (tùy chọn FastAPI / Streamlit)

---

## 4. Nội dung chi tiết từng notebook

### 01_prepare_dataset.ipynb
- Mục tiêu:
  + Kiểm tra và tiền xử lý dữ liệu
  + Tạo metadata (label, nhóm, ý nghĩa) cho mỗi ảnh
  + Chuẩn hóa định dạng và số lượng ảnh (nếu cần)

- Các bước chính:
  1. Duyệt các folder trong `./dataset_aug` để lấy danh sách ảnh theo loại biển
  2. Xây dựng DataFrame metadata: `image_path`, `label`, `group`, `meaning`, `class_id`
  3. Kiểm tra số lượng ảnh mỗi class, kiểm soát chất lượng ảnh bị lỗi
  4. Lưu metadata ra `metadata.csv` hoặc `dataset_metadata.parquet`

- Thư viện sử dụng:
  + os, pathlib, pandas, numpy, PIL, tqdm

- Input:
  + `dataset_aug/` có cấu trúc thư mục theo class

- Output:
  + `metadata.csv`
  + Bộ dữ liệu đã chuẩn (nếu cần copy/nén)

---

### 02_extract_embedding.ipynb
- Mục tiêu:
  + Tải mô hình CLIP pretrained (OpenAI CLIP / Hugging Face)
  + Trích xuất embedding vector cho mỗi ảnh
  + Lưu feature + metadata

- Các bước chính:
  1. Load metadata từ `metadata.csv`
  2. Load mô hình CLIP và tokenizer/processor
  3. Duyệt ảnh, tiền xử lý ảnh và tính embedding
  4. Chuẩn hóa embedding (L2 normalize)
  5. Lưu embedding kèm `image_path`, `label`, `group`, `meaning` (hdf5/numpy/pickle)

- Thư viện sử dụng:
  + torch, torchvision, transformers (hoặc clip), pandas, numpy, tqdm

- Input:
  + `metadata.csv`
  + `dataset_aug/`

- Output:
  + `image_embeddings.npy` (hoặc `embeddings.pkl`)
  + `embedding_metadata.csv`

---

### 03_build_vector_db.ipynb
- Mục tiêu:
  + Xây dựng chỉ mục vector search với FAISS hoặc Milvus
  + Lưu index ra disk

- Các bước chính:
  1. Load embedding file và metadata
  2. Chọn phương án vector store:
    - FAISS: IndexFlatIP / IndexIVFFlat + index.train/idx.add
    - Milvus: # tạo collection, load vector, gắn metadata
  3. Xây dựng index bằng cosine (nếu FAISS dùng dot product trên embedding chuẩn hóa)
  4. Lưu index: `faiss_index.faiss`, hoặc kết nối tới Milvus

- Thư viện sử dụng:
  + faiss-cpu (hoặc milvus-sdk-python), numpy, pandas

- Input:
  + `image_embeddings.npy`
  + `embedding_metadata.csv`

- Output:
  + `faiss_index.faiss`
  + `vector_metadata.csv`

---

### 04_search_demo.ipynb
- Mục tiêu:
  + Demo truy vấn ảnh (input ảnh, hoặc webcam)
  + Trả về top-K kết quả, nhóm biển và ý nghĩa

- Các bước chính:
  1. Load mô hình CLIP (ảnh query), load FAISS index + metadata
  2. Hàm `query_image(path)`:
    - Tính embedding CLIP của ảnh query
    - Chuẩn hóa
    - Search top-K (cosine)
    - Ghép metadata
  3. Hiển thị kết quả bằng matplotlib
  4. Thử với mẫu ảnh thuộc các nhóm: cấm/nguy hiểm/hiệu lệnh/chỉ dẫn

- Thư viện sử dụng:
  + matplotlib, PIL, numpy, pandas, torch, transformers, faiss

- Input:
  + `test_images/` (tạo folder đọc ảnh test)

- Output:
  + Báo cáo chính xác top-1/top-5 của sample demo

---

### 05_evaluation.ipynb
- Mục tiêu:
  + Tính toán độ chính xác thực nghiệm (top-1, top-5)
  + Tương tác đánh giá với tập kiểm thử nếu có

- Các bước chính:
  1. Chuẩn bị tập test (có label thực)
  2. Duyệt từng ảnh test, query vector DB và lấy top-5
  3. Tính top-1, top-5 accuracy
  4. In/bảng phân tích lỗi: nhóm nhầm lẫn, class ít ảnh, nhóm dễ nhầm
  5. Đề xuất cải tiến (SẮT ở phần bonus)

- Thư viện sử dụng:
  + scikit-learn, pandas, numpy, matplotlib, seaborn

- Input:
  + `metadata_test.csv` (hoặc tách từ metadata)
  + `faiss_index.faiss`

- Output:
  + `evaluation_report.md` hoặc `evaluation.json`

---

### 06_api_demo.ipynb (tùy chọn)
- Mục tiêu:
  + Triển khai API (FastAPI) hoặc demo UI (Streamlit)
  + Cho phép upload ảnh query và trả về kết quả trực quan

- Các bước chính:
  1. Cài dependencies: fastapi, uvicorn, pydantic, streamlit
  2. Viết endpoint `/predict`:
    - Nhận file ảnh
    - Tính embedding, search
    - Trả JSON: label/group/meaning/confidence/topK
  3. Tạo dashboard Streamlit: upload + show

- Output:
  + `app.py` hoặc Streamlit notebook

---

## 5. Pipeline tổng thể
1. Chuẩn bị dữ liệu + metadata
2. Trích xuất embedding bằng CLIP
3. Xây index vector bằng FAISS/Milvus
4. Thực hiện truy vấn từ ảnh mới
5. Trả kết quả tên biển, nhóm, ý nghĩa
6. Đánh giá với top-1/top-5
7. Tối ưu mở rộng (thêm class mới, incremental index update)

---

## 6. Cấu trúc thư mục dự án
```
TrafficSignsProject/
  README.md
  instruction.md
  dataset_aug/
  notebooks/
    01_prepare_dataset.ipynb
    02_extract_embedding.ipynb
    03_build_vector_db.ipynb
    04_search_demo.ipynb
    05_evaluation.ipynb
    06_api_demo.ipynb
  data/
    metadata.csv
    embedding_metadata.csv
    image_embeddings.npy
    faiss_index.faiss
  models/
    clip/
  outputs/
    evaluation_report.md
    demo_results/
  src/
    utils.py
    dataset.py
    clip_embedding.py
    vector_search.py
    api.py
```

## 7. Công nghệ sử dụng
- CLIP pretrained (OpenAI / Hugging Face): `ViT-B/32` hoặc `ViT-L/14`
- Vector DB:
  + FAISS (dễ triển khai, phù hợp dataset nhỏ)
  + Milvus (scale tốt, optional)
- Backend API: FastAPI (tùy chọn)
- Demo trực quan: Matplotlib / Streamlit

## 8. Yêu cầu nâng cao (giữ nguyên không training)
- Dùng chỉ model CLIP pretrained, không fine-tune
- Với dataset nhỏ ~3000 ảnh, dùng index CPU FAISS
- Mở rộng:
  + Khi thêm biển mới, chỉ cần tính embedding mới và `add` vào index
  + Update metadata csv/DB và không cần train lại model

## 9. Đánh giá và cải thiện
- Metrics:
  + Top-1 accuracy
  + Top-5 accuracy
  + Mean Average Precision (mAP) bổ sung (tuỳ ý)

- Cách đánh giá:
  + Tách `metadata` thành train/query/test
  + Với mỗi ảnh query, lấy top-5 và so sánh label
  + Top-1 correct nếu label đầu tiên trùng
  + Top-5 correct nếu label nằm trong top-5

- Gợi ý cải thiện:
  + Dùng ensemble từ text prompt nhóm (ví dụ query bằng "Biển cấm, Biển nguy hiểm"), so sánh clip image-text
  + Dùng augmentation thông minh (rotations, crop) khi lookup
  + Chuyển sang CLIP ViT-L/14 để embeddings mượt hơn
  + Dùng PQ hoặc HNSW nếu scale lớn hơn
  + Thêm pre-filter theo nhóm biển trước index để giảm nhầm lẫn

## 10. Lưu ý triển khai
- Dùng seed cố định để kết quả lặp lại
- Log tiến trình với tqdm
- Kiểm tra chất lượng ảnh (mất nét, ảnh lỗi) ở bước tiền xử lý
- Document rõ ràng trong mỗi notebook
- Mỗi notebook phải có phần "next step" ngắn gọn
