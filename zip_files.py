import os
from pathlib import Path
import zipfile


ROOT_DIR = 'C:/Users/PC/Documents/_mas/master rad/project'
SKIP = [
    'zip_files.py',
    'datasets',
]


def zip_with_skip(root_dir: str, skip: list[str]):
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f'{root} is not a valid folder')

    zip_path = root / f'{root.name}.zip'

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder_path, dirs, files in os.walk(root):
            # skip unwanted dirs
            dirs[:] = [d for d in dirs if d not in skip]

            for file in files:
                file_path = Path(folder_path) / file
                rel_path = file_path.relative_to(root)
                zipf.write(file_path, rel_path)

    print(f'Created: {zip_path} ({zip_path.stat().st_size / 1024:.2f} KB)')


zip_with_skip(ROOT_DIR, SKIP)
