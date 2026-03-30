import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import ViTImageProcessor, ViTModel

def load_vision_model(model_name='google/vit-base-patch16-224-in21k', device=None):
    """Load pretrained ViT model and processor (Cách 3.A)."""
    # Sử dụng mô hình ViT chuyên dụng cho phân tích hình ảnh thay vì CLIP
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    return model, processor, device


def load_image(image_path):
    """Open an image from disk as PIL image."""
    return Image.open(image_path).convert('RGB')


def get_image_embedding(image, processor, model, device='cpu'):
    """Get a normalized embedding vector for one PIL image (Cách 3.C)."""
    inputs = processor(images=image, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        # ViTModel trả về pooler_output là đại diện chuẩn cho toàn bộ bức ảnh
        # Code được làm sạch, không cần kiểm tra tuple/list phức tạp
        embedding = outputs.pooler_output 
        
        # Chuẩn hóa L2 để tìm kiếm Cosine Similarity trong FAISS
        normalized = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return normalized.cpu().numpy()[0]


def extract_image_embeddings(metadata_df, processor, model, device='cpu', batch_size=32):
    """Extract ViT image embeddings for all images referenced in metadata."""
    image_paths = metadata_df['image_path'].tolist()
    vectors = []
    hashed_paths = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc='Extracting embeddings'):
        batch_paths = image_paths[i:i + batch_size]
        images = []

        for p in batch_paths:
            try:
                images.append(load_image(p))
                hashed_paths.append(p)
            except Exception as e:
                print(f'Skipping corrupted image {p}: {e}')
                continue

        if len(images) == 0:
            continue

        inputs = processor(images=images, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Trích xuất trực tiếp pooler_output cho cả batch
            batch_embeddings = outputs.pooler_output
            
            # Chuẩn hóa L2
            batch_normalized = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            vectors.append(batch_normalized.cpu().numpy())

    if len(vectors) == 0:
        # Kích thước đầu ra của ViT-base là 768 chiều
        return np.empty((0, model.config.hidden_size), dtype=np.float32)

    embeddings = np.vstack(vectors).astype(np.float32)
    return embeddings


def save_embeddings(embeddings, output_npy_path):
    """Save embeddings as NumPy file."""
    np.save(output_npy_path, embeddings)


def load_embeddings(np_array_path):
    """Load embeddings NumPy file."""
    return np.load(np_array_path)