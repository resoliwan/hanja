"""Read ordered SCUT trajectories and SVG animation paths without rendering."""

import math
import json
import re
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from fontTools.misc.transform import Identity, Transform
from fontTools.pens.basePen import BasePen
from fontTools.svgLib.path import parse_path


def read_grade_entries(dictionary_root: Path, level: str, system='daehan-geomjeonghoe-hanja'):
    """Use the project's grade membership, not another exam's character list."""
    selected = []
    for path in sorted((dictionary_root / 'entries').glob('*/*.jsonc')):
        entry = json.loads(path.read_text(encoding='utf-8'))
        if any(g['systemId']==system and str(g['introducedLevel'])==str(level)
               for g in entry.get('gradeMemberships', [])):
            selected.append(dict(character=entry['headword'], entry_path=str(path.resolve()),
                                 declared_strokes=entry['strokes']['total'],
                                 current_asset=entry['strokes']['strokeOrder']['assetPath']))
    characters = [e['character'] for e in selected]
    if not selected or len(set(characters))!=len(characters) or any(len(c)!=1 for c in characters):
        raise ValueError('Grade selection must contain unique single-character entries')
    return selected


def read_scut(archive: Path, characters: str):
    """Preserve record offsets, original stroke order, and point order."""
    with zipfile.ZipFile(archive) as source:
        members = sorted(n for n in source.namelist() if n.endswith('.pot'))
        for member in members:
            subset = int(re.search(r' (\d+)\.pot$', member).group(1))
            labels_member = member.replace('/pot/', '/label/').replace('.pot', '.txt')
            labels = source.read(labels_member).decode('utf-8-sig').splitlines()
            data = source.read(member)
            offset = 0
            for record, label in enumerate(labels, 1):
                if offset + 8 > len(data):
                    raise ValueError(f'Truncated POT header: {member}:{record}')
                size, code, count = struct.unpack_from('<HIH', data, offset)
                end = offset + size
                decoded = code.to_bytes(4, 'little')[:2][::-1].decode('gbk')
                if size < 12 or end > len(data) or decoded != label:
                    raise ValueError(f'Invalid POT record: {member}:{record}')
                if label in characters:
                    strokes, stroke = [], []
                    cursor = offset + 8
                    terminated = False
                    while cursor + 4 <= end:
                        x, y = struct.unpack_from('<hh', data, cursor)
                        cursor += 4
                        if (x, y) == (-1, -1):
                            terminated = True
                            break
                        if (x, y) == (-1, 0):
                            if not stroke:
                                raise ValueError('Empty POT stroke')
                            strokes.append(stroke)
                            stroke = []
                        elif x < 0 or y < 0:
                            raise ValueError('Invalid POT coordinate')
                        else:
                            stroke.append([x, y])
                    if not terminated or cursor != end or stroke or len(strokes) != count:
                        raise ValueError(f'Invalid POT strokes: {member}:{record}')
                    yield dict(
                        sample_id=f'trad{subset}-{record:05d}', character=label,
                        subset=subset, record=record, byte_offset=offset,
                        source_member=member, strokes=strokes,
                    )
                offset = end
            if offset != len(data):
                raise ValueError(f'Unaccounted POT bytes: {member}')


class _PolylinePen(BasePen):
    """Flatten curves before arc-length resampling; reject disjoint strokes."""

    def __init__(self):
        super().__init__(None)
        self.points = []

    def _moveTo(self, point):
        if self.points:
            raise ValueError('A centerline must have one continuous subpath')
        self.points.append(point)

    def _lineTo(self, point):
        self.points.append(point)

    def _curveToOne(self, p1, p2, p3):
        p0 = self.points[-1]
        for i in range(1, 33):
            t, u = i / 32, 1 - i / 32
            self.points.append(tuple(
                u**3 * p0[k] + 3*u*u*t*p1[k] + 3*u*t*t*p2[k] + t**3*p3[k]
                for k in (0, 1)
            ))

    def _qCurveToOne(self, p1, p2):
        p0 = self.points[-1]
        for i in range(1, 33):
            t, u = i / 32, 1 - i / 32
            self.points.append(tuple(u*u*p0[k] + 2*u*t*p1[k] + t*t*p2[k] for k in (0, 1)))

    def _closePath(self):
        raise ValueError('Expected an open animation centerline, not an outline')

    def _endPath(self):
        pass


def _svg_transform(value):
    transform = Identity
    remainder = value
    for match in re.finditer(r'([A-Za-z]+)\s*\(([^)]*)\)', value):
        name, arguments = match.groups()
        numbers = [float(n) for n in re.findall(r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?', arguments)]
        if name == 'matrix' and len(numbers) == 6:
            local = Transform(*numbers)
        elif name == 'translate' and len(numbers) in (1, 2):
            local = Identity.translate(numbers[0], numbers[1] if len(numbers) == 2 else 0)
        elif name == 'scale' and len(numbers) in (1, 2):
            local = Identity.scale(numbers[0], numbers[-1])
        elif name == 'rotate' and len(numbers) in (1, 3):
            cx, cy = numbers[1:] if len(numbers) == 3 else (0, 0)
            local = Identity.translate(cx, cy).rotate(math.radians(numbers[0])).translate(-cx, -cy)
        else:
            raise ValueError(f'Unsupported SVG transform: {match.group()}')
        transform = transform.transform(local)
        remainder = remainder.replace(match.group(), '', 1)
    if remainder.strip(' ,\t\n'):
        raise ValueError(f'Invalid SVG transform: {value}')
    return transform


def read_svg_strokes(path: Path):
    """Select clip-path animation centerlines, not filled glyph outlines.

    SVG document order is not stroke order. Apply ancestor transforms before
    sorting by the numbered clip-path IDs. Animation durations are not pen time.
    """
    strokes = {}

    def visit(element, parent_transform):
        name = element.tag.rsplit('}', 1)[-1]
        if name in {'defs', 'script', 'style'}:
            return
        transform = parent_transform.transform(_svg_transform(element.get('transform', '')))
        if name == 'path' and element.get('clip-path'):
            match = re.fullmatch(r'url\(#[^()]*[cC](\d+)\)', element.get('clip-path'))
            if not match:
                raise ValueError(f'Unnumbered centerline: {element.get("clip-path")}')
            number = int(match.group(1))
            if number in strokes:
                raise ValueError(f'Duplicate stroke number: {number}')
            pen = _PolylinePen()
            parse_path(element.attrib['d'], pen)
            strokes[number] = [list(transform.transformPoint(p)) for p in pen.points]
        for child in element:
            visit(child, transform)

    visit(ET.parse(path).getroot(), Identity)
    if not strokes or sorted(strokes) != list(range(1, len(strokes) + 1)):
        raise ValueError(f'Missing or non-consecutive SVG strokes: {path}')
    return [strokes[number] for number in sorted(strokes)]
