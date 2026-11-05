"""Fetch a reproducible photo subset from the publicly distributed ReCTS ZIP."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import struct
import zlib
import zipfile

import requests
import urllib3
from paths import DATASETS

URL = 'https://datasets.cvc.uab.es/rrc/ReCTS.zip'


def _range(start, end):
    for attempt in range(3):
        try:
            # The public academic archive currently has an invalid TLS certificate.
            response = requests.get(URL, headers={'Range': f'bytes={start}-{end}'},
                                    timeout=60, verify=False)
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError('Archive server must support byte ranges')
            return response
        except requests.RequestException:
            if attempt == 2:
                raise


class _RemoteArchive(io.RawIOBase):
    def __init__(self):
        self.size = int(_range(0, 0).headers['Content-Range'].split('/')[-1])
        self.position = 0

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else size
        if not size:
            return b''
        content = _range(self.position, self.position + size - 1).content
        self.position += len(content)
        return content


def _download_member(info, output):
    destination = output / info.filename
    if destination.exists():
        content = destination.read_bytes()
        if len(content) == info.file_size and zlib.crc32(content) == info.CRC:
            return dict(path=info.filename, sha256=hashlib.sha256(content).hexdigest())
    header = _range(info.header_offset, info.header_offset + 29).content
    if header[:4] != b'PK\x03\x04':
        raise RuntimeError('Invalid ZIP local header')
    name_length, extra_length = struct.unpack_from('<HH', header, 26)
    start = info.header_offset + 30 + name_length + extra_length
    compressed = _range(start, start + info.compress_size - 1).content
    if info.compress_type == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(compressed, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        content = compressed
    else:
        raise ValueError('Unsupported archive compression')
    assert len(content) == info.file_size and zlib.crc32(content) == info.CRC
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_bytes(content)
    temporary.replace(destination)
    return dict(path=info.filename, sha256=hashlib.sha256(content).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DATASETS / 'rects-500')
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--exclude-manifest', type=Path)
    parser.add_argument('--purpose', default='evaluation only for our models')
    args = parser.parse_args()
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    args.output.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(_RemoteArchive())
    images = sorted(n for n in archive.namelist() if n.startswith('img/') and n.endswith('.jpg'))
    excluded = set()
    if args.exclude_manifest:
        excluded = set(json.loads(args.exclude_manifest.read_text())['selected_images'])
    eligible = [n for n in images if n not in excluded]
    selected = sorted(random.Random(args.seed).sample(eligible, args.count))
    assert not set(selected) & excluded
    members = [archive.getinfo(n) for image in selected
               for n in (image, 'gt_unicode/' + Path(image).stem + '.json')]
    design = dict(source=URL, split='official train; ' + args.purpose, excluded_images=len(excluded),
                  seed=args.seed, selected_images=selected, population_images=len(images),
                  sampling='uniform random photos, before inspecting labels or predictions',
                  tls_verification=False, integrity='ZIP CRC32 plus locally recorded SHA256')
    (args.output / 'selection.json').write_text(json.dumps(design, indent=2))
    print('SELECTED', len(selected), 'photos from', len(images), flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = []
        for record in executor.map(lambda info: _download_member(info, args.output), members):
            records.append(record)
            if len(records) % 100 == 0:
                print('DOWNLOADED', len(records), '/', len(members), flush=True)
    design['files'] = records
    (args.output / 'manifest.json').write_text(json.dumps(design, indent=2))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
