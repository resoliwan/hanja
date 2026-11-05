"""Compare the official Sabeom PDF with existing source label inventories.

NFC only merges Unicode canonical equivalents, never simplified/traditional
variants. Source presence is not an image-quality or recognition-accuracy test.
"""
import collections
import csv
import hashlib
import json
import re
import subprocess
import unicodedata
import zipfile

from paths import DATASETS, RESULTS


def canonical(character):
    return unicodedata.normalize('NFC', character)


def main():
    source = DATASETS / 'korean5000/official-grades-8-to-sabeom.pdf'
    output = RESULTS / 'sabeom5000-coverage'
    output.mkdir(exist_ok=True)
    text = subprocess.check_output(['pdftotext', '-raw', str(source), '-'], text=True)
    text = text.replace('[ㄱ,ㄴ,ㄷ順]', '')
    entries = re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]', text)
    assert len(entries) == 5000, len(entries)
    target = {canonical(c) for c in entries}
    ai_counts = collections.Counter()
    invalid = set()
    for name in (DATASETS / 'ai-free/archive-file-list.txt').read_text().splitlines():
        if not name.endswith('.png'):
            continue
        label = name.rsplit('/', 1)[-1].split('_')[0]
        if len(label) == 1:
            ai_counts[label] += 1
        else:
            invalid.add(label)
    scut_counts = collections.Counter()
    with zipfile.ZipFile(DATASETS / 'onHCCTestDB-TradChar.zip') as archive:
        for number in range(1, 6):
            name = f'onHCCTestDB-TradChar/label/onHCCTestDB-TradChar {number}.txt'
            scut_counts.update(archive.read(name).decode('utf-8-sig').splitlines())
    assert len(ai_counts) == 13063, len(ai_counts)
    assert len(scut_counts) == 5401, len(scut_counts)
    ai = {canonical(c) for c in ai_counts}
    scut = {canonical(c) for c in scut_counts}
    current = {canonical(c) for c in (DATASETS / 'korean-hanja3500-v1/classes.txt').read_text().splitlines()}
    def coverage(chars):
        return dict(total=len(chars), ai_free=len(chars & ai), scut=len(chars & scut),
                    either=len(chars & (ai | scut)), neither=len(chars - ai - scut))
    summary = dict(
        official_url='https://www.hanja.ne.kr/board/board/view.asp?tb=inno_12&num=15',
        pdf_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        official_entries=len(entries), exact_unique=len(set(entries)),
        canonical_unique=len(target), normalization='Unicode NFC; no variant substitution',
        duplicate_entries={c: n for c, n in collections.Counter(entries).items() if n > 1},
        canonical_collisions={c: sorted({e for e in entries if canonical(e) == c})
                              for c in target if len({e for e in entries if canonical(e) == c}) > 1},
        source_unique=dict(ai_free=len(ai_counts), scut=len(scut_counts)),
        malformed_ai_free_labels=sorted(invalid),
        official=coverage(target), additional_to_current=coverage(target-current),
        current_in_official=len(current & target), current_outside_official=sorted(current-target),
        combined_model=coverage(current | target),
        exact_codepoint_coverage=dict(ai_free=len(set(entries) & ai_counts.keys()),
                                      scut=len(set(entries) & scut_counts.keys())),
        limitation='Source label presence only. AI-FREE uses the saved archive inventory; SCUT reads all five label files. No pixel quality check or training performed. SCUT remains evaluation-only.',
    )
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    for name, chars in [('official-characters', target), ('additional-characters', target-current),
                        ('missing-both', target-ai-scut), ('current-outside-official', current-target)]:
        (output / f'{name}.txt').write_text('\n'.join(sorted(chars))+'\n')
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['character', 'codepoint', 'official_forms', 'in_current_model', 'in_ai_free', 'in_scut', 'missing_both'])
        for char in sorted(target):
            writer.writerow([char, f'U+{ord(char):04X}', ''.join(sorted({e for e in entries if canonical(e)==char})),
                             char in current, char in ai, char in scut, char not in ai | scut])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
