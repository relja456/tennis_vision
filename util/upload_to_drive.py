import os
from pathlib import Path

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


# --- CONFIG ---
LOCAL_FOLDER = Path(r'C:/Users/PC/Documents/_mas/master rad/project')
DRIVE_DEST = 'project_root'  # or "id:<folderId>"
SKIP_NAMES = {
    'datasets',  # whole folder
    'client_secrets.json',  # individual file
    'service_account.json',
    'pyproject.toml',
}
# -------------

FOLDER_MIME = 'application/vnd.google-apps.folder'


def auth_oauth():
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile('client_secrets.json')
    try:
        gauth.LocalWebserverAuth()
    except Exception:
        gauth.CommandLineAuth()
    return GoogleDrive(gauth)


def find_or_create_folder(drive, name, parent_id=None):
    q = f"mimeType='{FOLDER_MIME}' and trashed=false and title='{name}'"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    res = drive.ListFile({'q': q}).GetList()
    if res:
        return res[0]['id']
    meta = {'title': name, 'mimeType': FOLDER_MIME}
    if parent_id:
        meta['parents'] = [{'id': parent_id}]
    f = drive.CreateFile(meta)
    f.Upload()
    return f['id']


def ensure_dest(drive, dest):
    if dest.startswith('id:'):
        return dest[3:]
    parent = None
    for part in Path(dest).parts:
        parent = find_or_create_folder(drive, part, parent)
    return parent


def upload_tree(drive, root, dest_id):
    for r, dirs, files in os.walk(root):
        rel = Path(r).relative_to(root)

        # --- skip folder subtree if its name is in SKIP_NAMES
        if any(p in SKIP_NAMES for p in rel.parts):
            dirs[:] = []  # don't recurse further
            continue

        # mirror folder structure
        parent = dest_id
        for part in rel.parts:
            parent = find_or_create_folder(drive, part, parent)

        for name in files:
            if name in SKIP_NAMES:
                continue
            p = Path(r) / name
            gfile = drive.CreateFile({'title': p.name, 'parents': [{'id': parent}]})
            gfile.SetContentFile(str(p))
            gfile.Upload()
            print('[UP]', p)


if __name__ == '__main__':
    assert LOCAL_FOLDER.exists(), f'Missing {LOCAL_FOLDER}'
    drive = auth_oauth()
    dest_id = ensure_dest(drive, DRIVE_DEST)
    upload_tree(drive, LOCAL_FOLDER, dest_id)
    print('Done.')
