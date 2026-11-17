"""Behavior checks with known correspondence, independent of actual SCUT fits."""

import itertools
import math
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

from tools.handwriting.stroke_mapping.matcher import match_strokes, normalize_strokes
from tools.handwriting.stroke_mapping.sources import read_scut, read_svg_strokes


TEMPLATES = {
    '一': [[(0, 0.5), (1, 0.5)]],
    '二': [[(0.2, 0.2), (0.8, 0.2)], [(0, 0.8), (1, 0.8)]],
    '三': [[(0.1, 0.15), (0.9, 0.15)], [(0.2, 0.5), (0.8, 0.5)], [(0, 0.9), (1, 0.9)]],
}

YONG = [[(.1, .05), (.1, .95)],
        [(.1, .05), (.9, .05), (.9, .95), (.8, .9)],
        [(.1, .35), (.9, .35)], [(.1, .65), (.9, .65)],
        [(.5, .05), (.5, .95)]]


class StrokeMappingTests(unittest.TestCase):
    def test_all_writing_orders_and_directions(self):
        for character, target in TEMPLATES.items():
            for permutation in itertools.permutations(range(len(target))):
                for reversed_flags in itertools.product((False, True), repeat=len(target)):
                    source = [list(reversed(target[j])) if flag else target[j] for j, flag in zip(permutation, reversed_flags)]
                    source = [[(40 + 300*x, 70 + 300*y) for x, y in stroke] for stroke in source]
                    result = match_strokes(character, source, target)
                    with self.subTest(character=character, order=permutation, directions=reversed_flags):
                        self.assertEqual(result['status'], 'accepted', result['reasons'])
                        self.assertEqual([m['svg_stroke_number'] for m in result['mappings']], [j+1 for j in permutation])

    def test_uneven_point_density_does_not_move_center(self):
        sparse = [[(0, 0), (1, 0)]]
        dense = [[(0, 0), (.001, 0), (.002, 0), (.003, 0), (.9, 0), (1, 0)]]
        for a, b in zip(normalize_strokes(sparse)[0], normalize_strokes(dense)[0]):
            self.assertAlmostEqual(a[0], b[0])
            self.assertAlmostEqual(a[1], b[1])

    def test_bent_noisy_handwriting(self):
        target = TEMPLATES['三']
        source = [[(x, y + 0.02*x) for x, y in stroke] for stroke in target]
        source[1].insert(1, (0.5, 0.53))
        result = match_strokes('三', source, target)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual([m['svg_stroke_number'] for m in result['mappings']], [1, 2, 3])

    def test_extra_or_missing_strokes_are_not_forced(self):
        for source in (TEMPLATES['一'], TEMPLATES['三']):
            result = match_strokes('二', source, TEMPLATES['二'])
            self.assertEqual(result['mappings'], [])
            self.assertIn('stroke_count_mismatch', result['reasons'])

    def test_unique_mapping_is_separate_from_handwriting_quality(self):
        for stroke in [[(0, 0), (0, 1)], [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]:
            result = match_strokes('一', [stroke], TEMPLATES['一'])
            self.assertEqual(result['status'], 'accepted')
            self.assertEqual(result['correspondence_basis'], 'unique_given_label')
            self.assertTrue(result['quality_flags'])
            self.assertEqual(result['training_usage'], 'review_required')

    def test_order_alone_cannot_resolve_duplicate_strokes(self):
        result = match_strokes('二', [TEMPLATES['一'][0]] * 2, TEMPLATES['二'])
        self.assertEqual(result['status'], 'pending')
        self.assertIn('ambiguous_structure', result['reasons'])
        self.assertEqual(result['mappings'], [])

    def test_degenerate_or_nonfinite_input_is_pending(self):
        for stroke in [[(1, 1)], [(1, 1), (1, 1)], [(0, 0), (float('nan'), 1)]]:
            self.assertEqual(match_strokes('一', [stroke], TEMPLATES['一'])['status'], 'pending')

    def test_unsupported_character_fails_explicitly(self):
        with self.assertRaises(ValueError):
            match_strokes('金', TEMPLATES['三'], TEMPLATES['三'])

    def test_rotation_and_terminal_hook_do_not_block_correspondence(self):
        source = [list(s) for s in TEMPLATES['三']]
        source[0].append((.8, .23))
        angle = math.radians(-25)
        source = [[(x*math.cos(angle)-y*math.sin(angle), x*math.sin(angle)+y*math.cos(angle)) for x,y in s] for s in source]
        result = match_strokes('三', source, TEMPLATES['三'])
        self.assertEqual(result['status'], 'accepted', result['reasons'])
        self.assertEqual([m['svg_stroke_number'] for m in result['mappings']], [1, 2, 3])

    def test_yong_structure_with_reordered_and_reversed_strokes(self):
        for order in [(0,1,2,3,4), (4,3,2,1,0), (2,0,4,1,3), (1,3,0,4,2)]:
            angle = math.radians(-15)
            source = [[(70+240*(x*math.cos(angle)-y*math.sin(angle)),
                        90+240*(x*math.sin(angle)+y*math.cos(angle)))
                       for x,y in reversed(YONG[i])] for i in order]
            result = match_strokes('用', source, YONG)
            self.assertEqual(result['status'], 'accepted', result['reasons'])
            self.assertEqual([m['svg_stroke_number'] for m in result['mappings']], [i+1 for i in order])
            self.assertEqual(result['stability'], 1.0)

    def test_yong_duplicate_horizontals_are_ambiguous(self):
        source = list(YONG)
        source[3] = source[2]
        result = match_strokes('用', source, YONG)
        self.assertEqual(result['status'], 'pending')
        self.assertIn('ambiguous_structure', result['reasons'])
        self.assertEqual(result['mappings'], [])

    def test_expected_yong_corner_is_not_a_quality_problem(self):
        result = match_strokes('用', YONG, YONG)
        self.assertEqual(result['status'], 'accepted')
        self.assertNotIn('extra_bend_or_retrace', result['quality_flags'])
        self.assertEqual(result['training_usage'], 'candidate')


class SourceTests(unittest.TestCase):
    def test_svg_numbering_and_nested_transforms(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg"><g transform="scale(1,-1) translate(0,-100)">
        <path id="Xd1" d="M0 0L1 0L1 1Z"/>
        <path clip-path="url(#Xc2)" d="M10 20L90 20"/>
        <g transform="translate(5,0)"><path clip-path="url(#Xc1)" d="M10 80L70 80"/></g>
        </g></svg>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.svg'
            path.write_text(svg)
            self.assertEqual(read_svg_strokes(path), [[[15, 20], [75, 20]], [[10, 80], [90, 80]]])

    def test_svg_curves_and_invalid_numbering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.svg'
            path.write_text('<svg><path clip-path="url(#Xc1)" d="M0 0Q5 5 10 0"/></svg>')
            stroke = read_svg_strokes(path)[0]
            self.assertEqual(stroke[0], [0, 0])
            self.assertEqual(stroke[-1], [10, 0])
            self.assertGreater(max(p[1] for p in stroke), 0)
            path.write_text('<svg><path clip-path="url(#Xc2)" d="M0 0L1 1"/></svg>')
            with self.assertRaises(ValueError):
                read_svg_strokes(path)

    def test_pot_preserves_actual_order_and_validates_label(self):
        code = int.from_bytes('二'.encode('gbk')[::-1] + b'\0\0', 'little')
        points = [(10, 80), (90, 80), (-1, 0), (20, 20), (70, 20), (-1, 0), (-1, -1)]
        body = b''.join(struct.pack('<hh', *p) for p in points)
        data = struct.pack('<HIH', 8+len(body), code, 2) + body
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scut.zip'
            for label in ('二', '三'):
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr('test/pot/onHCCTestDB-TradChar 1.pot', data)
                    archive.writestr('test/label/onHCCTestDB-TradChar 1.txt', label+'\n')
                if label == '二':
                    row, = read_scut(path, '二')
                    self.assertEqual(row['strokes'], [[[10, 80], [90, 80]], [[20, 20], [70, 20]]])
                    self.assertEqual(row['byte_offset'], 0)
                else:
                    with self.assertRaises(ValueError):
                        list(read_scut(path, '二三'))


if __name__ == '__main__':
    unittest.main()
