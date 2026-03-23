import pandas as pd
from pathlib import Path

try:
    from .utils import is_image_file
except ImportError:
    from utils import is_image_file


GROUP_MAP = {
    'Prohibitory Signs': 'cấm',
    'Warning Signs': 'nguy hiểm',
    'Mandatory Signs': 'hiệu lệnh',
    'Information Signs': 'chỉ dẫn',
}


def scan_dataset(root_dir: str):
    """Scan dataset root containing class folders and build metadata DataFrame."""
    root = Path(root_dir)
    rows = []
    class_id = 0

    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name
        group_label = GROUP_MAP.get(group_name, group_name)

        for class_dir in sorted(group_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for img_path in sorted(class_dir.rglob('*')):
                if not img_path.is_file() or not is_image_file(img_path):
                    continue
                rows.append({
                    'image_path': str(img_path.resolve()),
                    'group': group_label,
                    'group_source': group_name,
                    'label': label,
                    'meaning': '',
                    'class_id': class_id,
                })
            class_id += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['group', 'label', 'image_path']).reset_index(drop=True)
    return df


def save_metadata(df, output_path):
    """Save metadata DataFrame to CSV."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8')


def load_metadata(csv_path):
    """Load metadata DataFrame from CSV."""
    return pd.read_csv(csv_path)
