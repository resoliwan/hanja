"""Bundle SimpleDic definitions without reordering readings or meanings."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('simpledic', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
characters = json.loads((root / 'apps/dic/assets/classes.json').read_text())
entries = {}
for path in sorted((args.simpledic / 'data/dictionaries/hanja-ko/entries').rglob('*.jsonc')):
    entry = json.loads(path.read_text())
    character = entry['headword']
    if character in entries:
        raise ValueError(f'Duplicate character: {character}')
    entries[character] = entry

if set(characters) != set(entries):
    raise ValueError('Dictionary characters do not match recognition classes')
output = root / 'apps/dic/assets/hanja_dictionary.json'
output.write_text(json.dumps(entries, ensure_ascii=False, separators=(',', ':')) + '\n')
print(f'Bundled {len(entries)} characters: {output}')
