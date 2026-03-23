# Vietnam Traffic Sign Retrieval

This project implements a traffic sign retrieval system using CLIP embeddings and FAISS for vector search. It allows users to upload images of traffic signs and retrieve the top-k most similar signs from a pre-built database.

## Features

- Image embedding extraction using CLIP (Contrastive Language-Image Pretraining)
- Vector search using FAISS (Facebook AI Similarity Search)
- FastAPI-based REST API for image retrieval
- Web interface for easy testing
- Jupyter notebooks for dataset preparation, embedding extraction, and evaluation

## Project Structure

- `src/`: Source code
  - `api.py`: FastAPI application for the retrieval service
  - `clip_embedding.py`: CLIP model loading and embedding extraction
  - `vector_search.py`: FAISS index operations
  - `dataset.py`: Dataset loading utilities
  - `utils.py`: Utility functions
- `data/`: Pre-built FAISS index, embeddings, and metadata
- `notebooks/`: Jupyter notebooks for various stages of the pipeline
- `dataset_aug/`: Augmented dataset with traffic sign images
- `requirements.txt`: Python dependencies

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd TrafficSignsProject
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API

To start the FastAPI server:

1. Navigate to the project root directory.

2. Run the API server:
   ```bash
   python src/api.py
   ```

   The server will start on `http://0.0.0.0:8000`.

3. Open your browser and go to `http://localhost:8000` to access the web interface.

4. Upload an image of a traffic sign and specify the number of top results (top-k) to retrieve similar signs.

## Testing with Notebooks

The project includes several Jupyter notebooks for testing and understanding the pipeline. Run them in the following order:

1. **01_prepare_dataset.ipynb**: Prepare and explore the dataset.

2. **02_extract_embedding.ipynb**: Extract CLIP embeddings from images.

3. **03_build_vector_db.ipynb**: Build the FAISS vector database.

4. **04_search_demo.ipynb**: Demonstrate vector search functionality.

5. **05_evaluation.ipynb**: Evaluate the retrieval performance.

6. **06_api_demo.ipynb**: Test the API endpoints.

To run the notebooks:

1. Ensure you have Jupyter installed:
   ```bash
   pip install jupyter
   ```

2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Navigate to the `notebooks/` directory and open the desired notebook.

4. Run the cells in order to execute the code.

## API Endpoints

- `GET /`: Home page with web interface
- `POST /predict`: Upload an image and get top-k predictions
- `GET /image?path=<image_path>`: Serve an image file for preview

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- FAISS
- FastAPI
- Uvicorn
- Other dependencies listed in `requirements.txt`

## License

[Add license information if applicable]