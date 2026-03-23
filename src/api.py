import os
import sys
try:
    import uvicorn
except ImportError:
    uvicorn = None

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from io import BytesIO

# Allow running as script: add src directory to path when needed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from .clip_embedding import load_clip_model, load_image, get_image_embedding
    from .vector_search import load_faiss_index, search_index
    from .dataset import load_metadata
except ImportError:
    from clip_embedding import load_clip_model, load_image, get_image_embedding
    from vector_search import load_faiss_index, search_index
    from dataset import load_metadata

import numpy as np
import torch

app = FastAPI(title='Vietnam Traffic Sign Retrieval')

@app.get('/', response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang='en'>
      <head>
        <meta charset='UTF-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1.0' />
        <title>Vietnam Traffic Sign Retrieval</title>
        <style>
          body { font-family: Arial, sans-serif; padding: 20px; max-width: 900px; margin: auto; }
          h1 { color: #1e3a8a; }
          .output { margin-top: 20px; }
          .result { border: 1px solid #ddd; border-radius: 6px; padding: 12px; margin-bottom: 10px; }
          .result span { font-weight: 600; }
          #loading { display: none; color: #0b74de; }
        </style>
      </head>
      <body>
        <h1>Vietnam Traffic Sign Retrieval</h1>
        <p>Upload an image of a traffic sign and get top-k nearest matches.</p>
        <label for='file'>Choose image:</label>
        <input type='file' id='file' accept='image/*' />
        <label for='top_k'>Top K:</label>
        <input type='number' id='top_k' value='5' min='1' max='20' style='width: 60px;' />
        <button id='predictBtn'>Predict</button>
        <span id='loading'>Processing...</span>
        <div>
          <strong>Input preview:</strong><br />
          <img id='inputPreview' alt='Input preview' style='max-width:300px; max-height:250px; border:1px solid #ccc; margin-top:10px;' />
        </div>
        <div class='output' id='output'></div>

        <script>
          const predictBtn = document.getElementById('predictBtn');
          const fileInput = document.getElementById('file');
          const output = document.getElementById('output');
          const inputPreview = document.getElementById('inputPreview');
          const loading = document.getElementById('loading');

          fileInput.addEventListener('change', () => {
            const file = fileInput.files[0];
            if (!file) {
              inputPreview.src = '';
              return;
            }
            inputPreview.src = URL.createObjectURL(file);
          });

          predictBtn.addEventListener('click', async () => {
            const file = fileInput.files[0];
            if (!file) {
              alert('Please choose an image file first.');
              return;
            }

            const topK = Math.max(1, Math.min(50, parseInt(document.getElementById('top_k').value || '5')));
            const formData = new FormData();
            formData.append('file', file);

            output.innerHTML = '';
            loading.style.display = 'inline';
            predictBtn.disabled = true;

            try {
              const resp = await fetch(`/predict?top_k=${topK}`, {
                method: 'POST',
                body: formData,
              });

              if (!resp.ok) {
                const errText = await resp.text();
                throw new Error(errText || `${resp.status} ${resp.statusText}`);
              }

              const data = await resp.json();
              if (!data.predictions || data.predictions.length === 0) {
                output.innerHTML = '<p>No predictions returned.</p>';
              } else {
                output.innerHTML = data.predictions.map(pred => {
                  const previewUrl = pred.image_path ? `/image?path=${encodeURIComponent(pred.image_path)}` : '';
                  return `
                  <div class='result'>
                    <div><span>Label:</span> ${pred.label || 'N/A'}</div>
                    <div><span>Group:</span> ${pred.group || 'N/A'}</div>
                    <div><span>Meaning:</span> ${pred.meaning || 'N/A'}</div>
                    <div><span>Score (distance):</span> ${pred.score?.toFixed(6) || 'N/A'}</div>
                    <div><span>Image path:</span> ${pred.image_path || 'N/A'}</div>
                    ${previewUrl ? `<div style='margin-top:8px;'><img src='${previewUrl}' style='max-width:220px; max-height:180px; border:1px solid #ddd;' alt='Candidate image' /></div>` : ''}
                  </div>`;
                }).join('');
              }
            } catch (error) {
              output.innerHTML = `<p style='color: red;'><strong>Error:</strong> ${error.message}</p>`;
            } finally {
              loading.style.display = 'none';
              predictBtn.disabled = false;
            }
          });
        </script>
      </body>
    </html>
    """

# References to loaded models/index to avoid repeated load in each request
_model = None
_processor = None
_device = None
_index = None
_metadata = None


def init_system(model_name='openai/clip-vit-base-patch32', index_path='data/faiss_index.faiss', metadata_path='data/metadata.csv', use_fast=False):
    global _model, _processor, _device, _index, _metadata

    if _model is None:
        _model, _processor, _device = load_clip_model(model_name=model_name, use_fast=use_fast)

    if _index is None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        _index = load_faiss_index(index_path)

    if _metadata is None:
        fallback_paths = ['data/metadata.csv', 'data/embedding_metadata.csv']
        final_path = metadata_path
        if not os.path.exists(final_path):
            for p in fallback_paths:
                if os.path.exists(p):
                    final_path = p
                    break
        if not os.path.exists(final_path):
            raise FileNotFoundError(
                f"Metadata CSV not found. Looked for: {metadata_path}, {', '.join(fallback_paths)}"
            )
        _metadata = load_metadata(final_path)


@app.get('/image')
async def get_image(path: str):
    # Serve image file for preview in UI (very basic, no auth)
    safe_path = os.path.normpath(path)
    if '..' in safe_path.replace('\\', '/').split('/'):
        raise HTTPException(status_code=400, detail='Invalid image path')
    if not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail='Image not found')
    return FileResponse(safe_path)


@app.post('/predict')
async def predict(file: UploadFile = File(...), top_k: int = 5):
    """Predict top-k traffic sign candidates from uploaded image."""
    global _model, _processor, _device, _index, _metadata

    if _model is None or _index is None or _metadata is None:
        init_system()

    image_data = await file.read()
    image = load_image(BytesIO(image_data)).convert('RGB')

    query_embedding = get_image_embedding(image, _processor, _model, device=_device)

    distances, indices = search_index(_index, query_embedding, top_k=top_k)

    import math

    def _sanitize_value(x):
        if isinstance(x, (float,)):
            if not math.isfinite(x):
                return None
            return float(x)
        if x is None:
            return None
        try:
            # JSON cannot encode NumPy scalars and can fail on NaN / inf
            return int(x) if isinstance(x, (int,)) else x
        except Exception:
            pass
        try:
            return float(x) if isinstance(x, (np.floating,)) else x
        except Exception:
            return x

    results = []
    for score, idx in zip(distances.tolist(), indices.tolist()):
        sanitized_score = None if not math.isfinite(score) else float(score)

        entry = _metadata.iloc[idx].to_dict()
        result = {
            'score': sanitized_score,
            'label': None if entry.get('label') is None else str(entry.get('label')),
            'group': None if entry.get('group') is None else str(entry.get('group')),
            'meaning': None if entry.get('meaning') is None else str(entry.get('meaning')),
            'image_path': None if entry.get('image_path') is None else str(entry.get('image_path')),
        }
        results.append(result)

    return JSONResponse({'predictions': results})


if __name__ == '__main__':
    init_system()
    uvicorn.run(app, host='0.0.0.0', port=8000)
