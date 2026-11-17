"""Grade selection follows the configured dictionary's exam and introduced level."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.handwriting.stroke_mapping.sources import read_grade_entries


class GradeSourceTests(unittest.TestCase):
    def test_uses_introduced_level_and_exam_system(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            entries = root/'entries'/'u4e'
            entries.mkdir(parents=True)
            for char, system, level in [('一','daehan-geomjeonghoe-hanja',8),
                                        ('二','daehan-geomjeonghoe-hanja',7),
                                        ('三','another-exam',8)]:
                entry = dict(headword=char, gradeMemberships=[dict(systemId=system,introducedLevel=level)],
                             strokes=dict(total=1,strokeOrder=dict(assetPath='assets/example.svg')))
                (entries/f'u{ord(char):04x}.jsonc').write_text(json.dumps(entry))
            self.assertEqual([e['character'] for e in read_grade_entries(root,'8')], ['一'])
            with self.assertRaises(ValueError):
                read_grade_entries(root,'6')


if __name__=='__main__':
    unittest.main()
