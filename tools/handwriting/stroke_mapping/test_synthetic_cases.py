"""Deterministic generation and labels independent of either matcher."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.handwriting.stroke_mapping.synthetic_cases import generate_svg_cases


class SyntheticCaseTests(unittest.TestCase):
    def test_ten_cases_are_repeatable_and_have_consistent_labels(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'u4e8c.svg'
            path.write_text('<svg xmlns="http://www.w3.org/2000/svg">'
                            '<path clip-path="url(#c1)" d="M 20 20 L 80 20"/>'
                            '<path clip-path="url(#c2)" d="M 0 80 L 100 80"/></svg>')
            cases=generate_svg_cases(path)
            self.assertEqual(len(cases),10)
            self.assertEqual(cases,generate_svg_cases(path))
            indexed={c['variant']:c for c in cases}
            self.assertEqual(indexed['split']['expected_relations'],[[1,1],[2,2],[3,2]])
            self.assertEqual(indexed['join']['expected_relations'],[[1,1],[1,2]])
            self.assertIn(None,indexed['join']['expected_edge_labels'][0])
            self.assertEqual(indexed['duplicate']['expected_relations'],[[1,1],[2,2],[3,1]])
            self.assertNotEqual(indexed['duplicate']['strokes'][0],indexed['duplicate']['strokes'][-1])
            self.assertEqual(indexed['duplicate']['generator_version'],2)
            self.assertEqual(indexed['extra_ink']['expected_relations'],[[1,1],[2,2]])
            self.assertEqual(indexed['reverse_order']['expected_relations'],[[1,2],[2,1]])
            self.assertEqual(indexed['different_character']['evaluation_policy'],'whole_character_rejection')
            for case in cases:
                for points,labels in zip(case['strokes'],case['expected_edge_labels']):
                    self.assertEqual(len(points)-1,len(labels))
            cases[0]['strokes'][0][0][0]=-100
            self.assertNotEqual(cases[0]['strokes'],cases[1]['strokes'])
            self.assertEqual(generate_svg_cases(path),generate_svg_cases(path))
