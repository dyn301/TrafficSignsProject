# TAI LIEU TONG QUAN HE THONG VA HUONG DAN DEMO

## 1) Muc tieu de tai

He thong duoc xay dung de tra cuu bien bao giao thong Viet Nam theo anh dau vao.  
Nguoi dung upload 1 anh bien bao, he thong tra ve top-k anh trong co so du lieu co do tuong dong cao nhat, kem thong tin:
- ma bien bao (`label`)
- nhom bien bao (`group`)
- y nghia (`meaning`)
- diem tuong dong (`score`)

Huong tiep can hien tai la **image retrieval** (tim anh giong nhat) thay vi classifier co dinh.

## 2) Tong quan kien truc he thong

He thong gom 4 lop chinh:

1. **Du lieu va metadata**
   - Danh sach anh + nhan duoc luu trong `data/metadata.csv`.
   - Anh duoc to chuc theo nhom trong `dataset_aug/`.

2. **Sinh vector dac trung (embedding)**
   - Dung CLIP (`openai/clip-vit-base-patch32`) trong `src/clip_embedding.py`.
   - Moi anh -> 1 vector embedding da chuan hoa L2.

3. **Tim kiem tuong dong**
   - Dung FAISS trong `src/vector_search.py`.
   - Chi so `IndexFlatIP` (inner product) + vector da normalize -> xap xi cosine similarity.

4. **API + giao dien demo**
   - FastAPI trong `src/api.py`.
   - Endpoint du doan: `POST /predict?top_k=...`.
   - Trang web demo tai `GET /` (upload anh, chon K, hien thi ket qua card).

## 3) Luong xu ly end-to-end

1. Quet dataset va tao metadata (`notebooks/01_prepare_dataset.ipynb`, `src/dataset.py`).
2. Trich xuat embedding CLIP (`notebooks/02_extract_embedding.ipynb`).
3. Build FAISS index (`notebooks/03_build_vector_db.ipynb`).
4. Chay API (`src/api.py`), upload anh tren web UI.
5. API:
   - doc anh upload
   - sinh query embedding
   - search FAISS top-k
   - map index -> metadata
   - tra JSON de giao dien hien thi.

## 4) Cau truc code da thuc hien

### `src/clip_embedding.py`
- Load model CLIP + processor.
- Ham lay embedding cho 1 anh (`get_image_embedding`).
- Ham trich xuat embedding hang loat (`extract_image_embeddings`).
- Co xu ly bo qua anh loi/corrupted khi batch.

### `src/vector_search.py`
- Tao index FAISS (`build_faiss_index`).
- Luu/nap index (`faiss.write_index`, `faiss.read_index`).
- Tim kiem top-k (`search_index`), co check kich thuoc vector.

### `src/dataset.py`
- Quet folder dataset va tao bang metadata.
- Anh xa nhom bien bao Anh -> Viet (`GROUP_MAP`).
- Luu/nap metadata CSV.

### `src/api.py`
- Khoi tao FastAPI + giao dien HTML/Tailwind nhung trong API.
- Lazy init model/index/metadata de tranh load lai moi request.
- Endpoint:
  - `GET /`: trang demo upload.
  - `POST /predict`: tra top-k ket qua.
  - `GET /image?path=...`: tra file anh de preview ket qua.
- Co sanitize gia tri JSON va check path co ban.

## 5) Hien trang du lieu va tai nguyen

- `metadata.csv` hien co:
  - ~3000 anh
  - 300 lop bien bao (`class_id`)
  - 4 nhom chinh: `cam`, `nguy hiem`, `hieu lenh`, `chi dan`
- Trong `data/` hien tai dang co file metadata, can tao them:
  - FAISS index (`faiss_index.faiss`)
  - (tuy chon) embedding `.npy`

## 6) Cach demo he thong cho giang vien

## 6.1 Chuan bi truoc demo (mot lan)

1. Cai thu vien:
```bash
pip install -r requirements.txt
```

2. Tao embedding va FAISS index (neu chua co):
   - Chay lan luot notebook:
     - `notebooks/01_prepare_dataset.ipynb`
     - `notebooks/02_extract_embedding.ipynb`
     - `notebooks/03_build_vector_db.ipynb`
   - Dam bao sinh ra file index tai `data/faiss_index.faiss` (hoac cap nhat duong dan trong `init_system`).

## 6.2 Demo truc tiep tren lop

1. Chay API:
```bash
python src/api.py
```

2. Mo trinh duyet: `http://localhost:8000`

3. Kich ban trinh bay de duoc danh gia tot:
   - **Case 1 - Anh ro, dung bien**: ket qua top-1 dung nhan va y nghia.
   - **Case 2 - Anh thay doi goc/chieu sang**: top-3 van chua bien dung.
   - **Case 3 - Nhieu bien gan nhau**: so sanh score giua ket qua dung va ket qua gan dung.
   - **Case 4 - Dieu chinh K**: tang/giảm `top_k` de thay do on dinh retrieval.

4. Giai thich cho giang vien khi demo:
   - He thong khong chi "phan lop cung", ma tra nhieu ung vien co score.
   - Uu diem voi du lieu thuc te: linh hoat hon khi anh bi bien doi nhe.

## 7) Danh gia nhung mat da lam duoc

## 7.1 Ky thuat da hoan thanh
- Da hoan thanh pipeline retrieval tu du lieu -> embedding -> vector DB -> API.
- Da trien khai duoc web UI de demo truc tiep, than thien cho nguoi danh gia.
- Da tach module ro rang (`dataset`, `embedding`, `search`, `api`) de de bao tri.
- Da co metadata gom ma bien + y nghia, phu hop bai toan nghiep vu.

## 7.2 Gia tri hoc thuat va ung dung
- Chon huong retrieval bang CLIP + FAISS phu hop bai toan tim anh tuong dong.
- Co kha nang mo rong sang tim kiem theo mo ta van ban (nhung chua trien khai day du).
- Co the su dung lam nen tang cho tro giup hoc bien bao giao thong.

## 8) Han che hien tai

- Chua co bo danh gia dinh luong ro rang (mAP, Recall@K, Precision@K theo tung nhom).
- Duong dan anh trong metadata dang la absolute path cua may phat trien, tinh di dong chua cao.
- `GET /image` moi moi o muc check co ban, chua co co che auth/cache.
- Chua toi uu toc do cho quy mo lon hon (hang chuc/hang tram nghin anh).
- Chua dong goi deployment (Docker, script setup 1 lenh, monitor/log chuan).

## 9) Huong phat trien tiep theo (giu cong nghe hien tai)

Van giu stack Python + CLIP + FAISS + FastAPI, uu tien cac huong sau:

1. **Nang chat luong retrieval**
   - Bo sung augmentation co chu dich.
   - Lam sach metadata va bo sung `meaning` day du, nhat quan.
   - Fine-tune CLIP nhe tren tap bien bao Viet Nam (neu co GPU/thoi gian).

2. **Danh gia he thong bai ban**
   - Them notebook/skript tinh Recall@1/5/10, mAP.
   - Bao cao ket qua theo tung nhom bien bao de tim diem yeu.

3. **Toi uu he thong tim kiem**
   - Chuyen tu `IndexFlatIP` sang IVF/HNSW khi du lieu lon hon.
   - Them buoc re-ranking top-N de cai thien top-1.

4. **Nang cap API va UI**
   - Them endpoint healthcheck/version.
   - Bo sung luu lich su truy van + anh input de demo thong ke.
   - Hien thi ly do goi y (score, nhom, mo ta) ro hon.

5. **San sang trien khai**
   - Dong goi Docker + file cau hinh duong dan du lieu.
   - Chuan hoa logging, theo doi loi va huong dan van hanh.

## 10) De xuat lo trinh ngan han (2-4 tuan)

- Tuan 1: Chuan hoa du lieu, metadata, script tao index tu dong.
- Tuan 2: Viet bo danh gia dinh luong + bao cao baseline.
- Tuan 3: Toi uu FAISS va test benchmark toc do/chinh xac.
- Tuan 4: Hoan thien demo script cho giang vien + tai lieu deployment.

---

Neu can, co the tao them 1 file "script thuyet trinh demo 5-7 phut" de khi bao cao chi can bam theo tung buoc va noi dung da chuan hoa.
