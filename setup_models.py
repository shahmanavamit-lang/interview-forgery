"""Download the pinned local English speech model; no interview data is sent."""
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = 'vosk-model-small-en-us-0.15'
URL = f'https://alphacephei.com/vosk/models/{NAME}.zip'
SHA256 = '30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498'


def main():
    target = ROOT / 'models' / NAME
    if (target / 'am' / 'final.mdl').is_file():
        print('Local speech model already installed.')
        return
    if target.exists():
        raise SystemExit('Incomplete model directory exists. Move it aside and run setup again.')
    target.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='p1-model-', dir=target.parent) as folder:
        temp = Path(folder)
        archive = temp / 'speech.zip'
        digest = hashlib.sha256()
        size = 0
        print('Downloading local speech model (about 40 MB)…')
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 60_000_000:
                    raise ValueError('Download exceeded expected size')
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != SHA256:
            raise ValueError('Model checksum did not match; nothing was installed')
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist()) > 150_000_000:
                raise ValueError('Expanded model exceeds size limit')
            for entry in z.infolist():
                path = (temp / entry.filename).resolve()
                if not path.is_relative_to(temp.resolve()) or not entry.filename.startswith(NAME + '/'):
                    raise ValueError('Unsafe archive path')
            z.extractall(temp)
        shutil.move(str(temp / NAME), target)
    print('Local speech model installed.')


if __name__ == '__main__':
    main()
