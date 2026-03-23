import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel


def load_clip_model(model_name='openai/clip-vit-base-patch32', device=None, use_fast=False):
    """Load pretrained CLIP model and processor."""
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=use_fast)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    return model, processor, device


def load_image(image_path):
    """Open an image from disk as PIL image."""
    return Image.open(image_path).convert('RGB')


def get_image_embedding(image, processor, model, device='cpu'):
    """Get a normalized embedding vector for one PIL image."""
    inputs = processor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        image_outputs = model.get_image_features(**inputs)

        if hasattr(image_outputs, 'pooler_output') and image_outputs.pooler_output is not None:
            embedding = image_outputs.pooler_output
        elif isinstance(image_outputs, (tuple, list)) and len(image_outputs) > 0:
            embedding = image_outputs[0]
        else:
            embedding = image_outputs

        if not torch.is_tensor(embedding):
            raise TypeError(f'Unexpected embedding type: {type(embedding)}')

        normalized = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return normalized.cpu().numpy()[0]


def extract_image_embeddings(metadata_df, processor, model, device='cpu', batch_size=32):
    """Extract CLIP image embeddings for all images referenced in metadata."""
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
                # Skip unreadable/unsupported images and continue.
                print(f'Skipping corrupted image {p}: {e}')
                continue

        if len(images) == 0:
            continue

        inputs = processor(images=images, return_tensors='pt').to(device)

        with torch.no_grad():
            image_outputs = model.get_image_features(**inputs)
            # CLIP get_image_features may return a BaseModelOutputWithPooling or tensor
            if hasattr(image_outputs, 'pooler_output') and image_outputs.pooler_output is not None:
                image_features = image_outputs.pooler_output
            elif isinstance(image_outputs, (tuple, list)) and len(image_outputs) > 0:
                image_features = image_outputs[0]
            else:
                image_features = image_outputs

            if not torch.is_tensor(image_features):
                raise TypeError(f'Unexpected image_features type: {type(image_features)}')

            # L2 normalize as standard for cosine similarity with dot product
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            vectors.append(image_features.cpu().numpy())

    if len(vectors) == 0:
        return np.empty((0, model.visual_projection.out_features), dtype=np.float32)

    embeddings = np.vstack(vectors).astype(np.float32)
    return embeddings


def save_embeddings(embeddings, output_npy_path):
    """Save embeddings as NumPy file."""
    np.save(output_npy_path, embeddings)


def load_embeddings(np_array_path):
    """Load embeddings NumPy file."""
    return np.load(np_array_path)
