"""Fetch pinned fonts and AI-FREE archive parts, retaining source licenses."""
import concurrent.futures, hashlib, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / 'data/fonts-korean-hanja'
AI_DIR = ROOT / 'data/ai-free'

def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)

def download(url, path, size=None, blob_sha=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or (size and path.stat().st_size != size):
        temp = path.with_suffix(path.suffix + '.part')
        with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if size:
            assert temp.stat().st_size == size, path
        temp.replace(path)
    data = path.read_bytes()
    if blob_sha:
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == blob_sha, path
    print('Verified', path.name, len(data), flush=True)
    return {'url': url, 'path': str(path.relative_to(ROOT)), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(), 'git_blob_sha1': blob_sha}

def main():
    specs = [
        ('noto-sans-kr', 'notofonts/noto-cjk', 'Sans/OTF/Korean/NotoSansCJKkr-Regular.otf', 'Sans/LICENSE'),
        ('noto-serif-kr', 'notofonts/noto-cjk', 'Serif/OTF/Korean/NotoSerifCJKkr-Regular.otf', 'Serif/LICENSE'),
        ('un-batang', 'deepin-community/fonts-unfonts-core', 'UnBatang.ttf', 'COPYING'),
        ('un-dotum', 'deepin-community/fonts-unfonts-core', 'UnDotum.ttf', 'COPYING'),
        ('lxgw-wenkai-kr', 'lxgw/LxgwWenkaiKR', 'fonts/TTF/LXGWWenKaiKR-Regular.ttf', 'OFL.txt'),
    ]
    revisions = {
        'notofonts/noto-cjk': 'f8d157532fbfaeda587e826d4cd5b21a49186f7c',
        'deepin-community/fonts-unfonts-core': 'ef6c68f4f0beb2a03fcf460aa643d73389aac59a',
        'lxgw/LxgwWenkaiKR': 'dd9ce0bb2f31d40c9a965b2c86527e79af329f62',
    }
    def fetch_font(spec):
        name, repo, location, license_path = spec
        base = f'https://raw.githubusercontent.com/{repo}/{revisions[repo]}/'
        font = download(base + location, FONT_DIR / name / Path(location).name)
        lic = download(base + license_path, FONT_DIR / name / 'LICENSE.txt')
        return dict(id=name, repository=repo, revision=revisions[repo], font=font, license=lic)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        fonts = list(pool.map(fetch_font, specs))
    (FONT_DIR / 'sources.json').write_text(json.dumps(fonts, indent=2))
    repo = 'chenkenanalytic/handwritting_data_all'
    rev = '2388ea0be27a3b966994c094e0b7fff0357deea9'
    tree = get_json(f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1')['tree']
    items = [item for item in tree if item['path'].startswith('all_data.zip.')]
    base = f'https://raw.githubusercontent.com/{repo}/{rev}/'
    def fetch_part(item):
        return download(base + item['path'], AI_DIR / item['path'], item['size'], item['sha'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(fetch_part, items))
    readme = download(base + 'README.md', AI_DIR / 'README-pinned.md')
    (AI_DIR / 'download-manifest.json').write_text(json.dumps(dict(repository=repo, revision=rev, parts=parts, readme=readme), indent=2))
    print('ALL SOURCES READY', flush=True)

if __name__ == '__main__':
    main()
