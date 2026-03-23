import os
from pathlib import Path


def ensure_dir(path):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)


def is_image_file(path):
    """Check whether a file path is an image by extension."""
    return str(path).lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))


def normalize_text(s):
    """Normalize text label to safe form."""
    return str(s).strip().replace(' ', '_').lower()