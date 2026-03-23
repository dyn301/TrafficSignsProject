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
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vietnam Traffic Sign Retrieval</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .glass-morphism {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: #f1f1f1; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #888; border-radius: 10px; }
        .result-card:hover { transform: translateY(-4px); transition: all 0.3s ease; }
    </style>
</head>
<body class="bg-slate-50 text-slate-900 min-h-screen">

    <!-- Header Section -->
    <header class="bg-white border-b sticky top-0 z-50">
        <div class="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
            <div class="flex items-center gap-2">
                <div class="bg-blue-600 p-2 rounded-lg">
                    <i data-lucide="shield-alert" class="text-white w-6 h-6"></i>
                </div>
                <h1 class="text-xl font-bold tracking-tight text-slate-800">VN Traffic Sign <span class="text-blue-600">Retrieval</span></h1>
            </div>
        </div>
    </header>

    <main class="max-w-6xl mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        <!-- Left Column: Controls & Preview -->
        <div class="lg:col-span-4 space-y-6">
            <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
                    <i data-lucide="upload" class="w-5 h-5"></i> Input Image
                </h2>
                
                <!-- Upload Area -->
                <div id="dropzone" class="relative border-2 border-dashed border-slate-300 rounded-xl p-4 flex flex-col items-center justify-center gap-3 bg-slate-50 hover:bg-slate-100 transition-colors cursor-pointer group mb-4">
                    <input type="file" id="file" accept="image/*" class="absolute inset-0 opacity-0 cursor-pointer" />
                    <div id="uploadPlaceholder" class="text-center py-4">
                        <i data-lucide="image" class="w-10 h-10 text-slate-400 mx-auto mb-2 group-hover:scale-110 transition-transform"></i>
                        <p class="text-sm font-medium text-slate-600">Nhấp hoặc kéo thả để tải ảnh</p>
                        <p class="text-xs text-slate-400">JPG, PNG lên đến 10MB</p>
                    </div>
                    <img id="inputPreview" class="hidden w-full h-auto max-h-64 object-contain rounded-lg shadow-sm" alt="Input preview" />
                    <button id="clearBtn" class="hidden absolute top-2 right-2 p-1 bg-white/80 rounded-full hover:bg-red-50 text-red-500 shadow-sm border transition-colors">
                        <i data-lucide="x" class="w-4 h-4"></i>
                    </button>
                </div>

                <!-- Parameters -->
                <div class="space-y-4">
                    <div>
                        <label for="top_k" class="block text-sm font-medium text-slate-700 mb-1">Số kết quả hàng đầu (K)</label>
                        <div class="flex items-center gap-3">
                            <input type="range" id="top_k_range" min="1" max="20" value="5" class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600" />
                            <input type="number" id="top_k" value="5" min="1" max="20" class="w-16 border rounded-lg px-2 py-1 text-sm font-semibold text-center focus:ring-2 focus:ring-blue-500 outline-none" />
                        </div>
                    </div>

                    <button id="predictBtn" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-xl shadow-lg shadow-blue-200 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed">
                        <span id="btnText">Tìm kiếm kết quả</span>
                        <i data-lucide="search" class="w-5 h-5"></i>
                        <div id="loader" class="hidden w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    </button>
                </div>
            </div>

            <div id="statusMessage" class="hidden p-4 rounded-xl text-sm border"></div>
        </div>

        <!-- Right Column: Results -->
        <div class="lg:col-span-8">
            <div class="flex items-center justify-between mb-6">
                <h2 class="text-xl font-bold text-slate-800">Kết quả tìm kiếm</h2>
                <div id="resultStats" class="text-sm text-slate-500"></div>
            </div>

            <!-- Empty State -->
            <div id="emptyState" class="bg-white border-2 border-dashed border-slate-200 rounded-2xl p-12 text-center">
                <div class="bg-slate-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i data-lucide="scan-search" class="w-8 h-8 text-slate-400"></i>
                </div>
                <h3 class="text-lg font-medium text-slate-700">Chưa có kết quả</h3>
                <p class="text-slate-500 text-sm max-w-xs mx-auto">Tải ảnh lên và nhấn nút tìm kiếm để xem các biển báo tương ứng.</p>
            </div>

            <!-- Grid Output -->
            <div id="output" class="grid grid-cols-1 sm:grid-cols-2 gap-4"></div>
        </div>
    </main>

    <script>
        // Initialize Icons
        lucide.createIcons();

        const fileInput = document.getElementById('file');
        const topKInput = document.getElementById('top_k');
        const topKRange = document.getElementById('top_k_range');
        const predictBtn = document.getElementById('predictBtn');
        const btnText = document.getElementById('btnText');
        const loader = document.getElementById('loader');
        const inputPreview = document.getElementById('inputPreview');
        const uploadPlaceholder = document.getElementById('uploadPlaceholder');
        const clearBtn = document.getElementById('clearBtn');
        const output = document.getElementById('output');
        const emptyState = document.getElementById('emptyState');
        const statusMessage = document.getElementById('statusMessage');
        const resultStats = document.getElementById('resultStats');

        // Sync Range and Number Inputs
        topKRange.addEventListener('input', () => topKInput.value = topKRange.value);
        topKInput.addEventListener('input', () => topKRange.value = topKInput.value);

        // Preview image logic
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const url = URL.createObjectURL(file);
                inputPreview.src = url;
                inputPreview.classList.remove('hidden');
                uploadPlaceholder.classList.add('hidden');
                clearBtn.classList.remove('hidden');
            }
        });

        clearBtn.addEventListener('click', () => {
            fileInput.value = '';
            inputPreview.src = '';
            inputPreview.classList.add('hidden');
            uploadPlaceholder.classList.remove('hidden');
            clearBtn.classList.add('hidden');
            output.innerHTML = '';
            emptyState.classList.remove('hidden');
            statusMessage.classList.add('hidden');
            resultStats.textContent = '';
        });

        function showStatus(msg, type = 'error') {
            statusMessage.textContent = msg;
            statusMessage.classList.remove('hidden', 'bg-red-50', 'border-red-200', 'text-red-700', 'bg-blue-50', 'border-blue-200', 'text-blue-700');
            if (type === 'error') {
                statusMessage.classList.add('bg-red-50', 'border-red-200', 'text-red-700');
            } else {
                statusMessage.classList.add('bg-blue-50', 'border-blue-200', 'text-blue-700');
            }
        }

        predictBtn.addEventListener('click', async () => {
            const file = fileInput.files[0];
            if (!file) {
                showStatus('Vui lòng chọn hoặc kéo thả một hình ảnh.');
                return;
            }

            const k = topKInput.value;
            const formData = new FormData();
            formData.append('file', file);

            // UI State loading
            loader.classList.remove('hidden');
            btnText.textContent = 'Đang tìm...';
            predictBtn.disabled = true;
            statusMessage.classList.add('hidden');

            try {
                const resp = await fetch(`/predict?top_k=${k}`, {
                    method: 'POST',
                    body: formData,
                });

                if (!resp.ok) throw new Error(await resp.text() || 'Lỗi hệ thống');

                const data = await resp.json();
                
                if (!data.predictions || data.predictions.length === 0) {
                    output.innerHTML = '';
                    emptyState.classList.remove('hidden');
                    resultStats.textContent = 'Không tìm thấy kết quả';
                } else {
                    emptyState.classList.add('hidden');
                    resultStats.textContent = `Tìm thấy ${data.predictions.length} kết quả`;
                    
                    output.innerHTML = data.predictions.map((pred, index) => {
                        const previewUrl = pred.image_path ? `/image?path=${encodeURIComponent(pred.image_path)}` : '';
                        const rankColor = index === 0 ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-600';
                        
                        return `
                        <div class="result-card bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden group flex flex-col h-full">
                            <div class="relative h-44 bg-slate-50 overflow-hidden shrink-0">
                                ${previewUrl ? 
                                    `<img src="${previewUrl}" class="w-full h-full object-contain p-4 group-hover:scale-110 transition-transform duration-500" alt="Sign" onerror="this.src='https://via.placeholder.com/300?text=No+Image'"/>` 
                                    : `<div class="w-full h-full flex items-center justify-center"><i data-lucide="image-off" class="text-slate-300 w-12 h-12"></i></div>`
                                }
                                <div class="absolute top-3 left-3 px-3 py-1 rounded-full text-[10px] font-bold shadow-sm ${rankColor}">
                                    #${index + 1} Tương đồng
                                </div>
                                <div class="absolute top-3 right-3 bg-white/90 backdrop-blur px-2 py-1 rounded-lg text-[9px] font-mono border border-slate-100 shadow-sm">
                                    Score: ${pred.score?.toFixed(4) || '0.000'}
                                </div>
                            </div>
                            <div class="p-4 flex flex-col flex-grow space-y-3">
                                <div class="flex-grow">
                                    <div class="text-[9px] uppercase font-bold tracking-widest text-blue-600 mb-0.5">${pred.group || 'CHƯA PHÂN LOẠI'}</div>
                                    <h3 class="text-base font-bold text-slate-800 leading-snug mb-2">${pred.label || 'Biển báo không xác định'}</h3>
                                    <p class="text-sm text-slate-600 leading-relaxed italic border-l-2 border-slate-100 pl-3">
                                        "${pred.meaning || 'Không có mô tả chi tiết cho biển báo này.'}"
                                    </p>
                                </div>
                                
                                <div class="pt-3 border-t border-slate-50 mt-auto">
                                    <div class="flex items-center gap-2 text-[9px] text-slate-400">
                                        <i data-lucide="file-code" class="w-3 h-3 text-slate-300"></i>
                                        <span class="truncate">${pred.image_path || 'unknown/path'}</span>
                                    </div>
                                </div>
                            </div>
                        </div>`;
                    }).join('');
                    
                    // Re-render icons for new elements
                    lucide.createIcons();
                }

            } catch (error) {
                showStatus(error.message);
                emptyState.classList.remove('hidden');
            } finally {
                loader.classList.add('hidden');
                btnText.textContent = 'Tìm kiếm kết quả';
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
