"""Known many-to-many examples, gaps, crossing rejection and source locations."""

import unittest

from tools.handwriting.stroke_mapping.segment_matcher import (
    match_segments, _union_length, _sample_aligned, _source_location,
)


ONE = [[(0,.5),(1,.5)]]
TWO = [[(.2,.2),(.8,.2)],[(0,.8),(1,.8)]]
YONG = [[(.1,.05),(.1,.95)],[(.1,.05),(.9,.05),(.9,.95)],
        [(.1,.35),(.9,.35)],[(.1,.65),(.9,.65)],[(.5,.05),(.5,.95)]]


def labels(result, number):
    return {r['svg_stroke_number'] for r in result['proposed_mappings'] if r['source_stroke_number']==number}


class SegmentMappingTests(unittest.TestCase):
    def test_single_input_can_match_two_template_strokes(self):
        result = match_segments('二', [TWO[0]+TWO[1]], TWO)
        self.assertEqual(labels(result,1), {1,2})
        self.assertEqual(result['status'],'accepted',result['target_coverage'])
        self.assertTrue(any(r['kind']=='connector_candidate' for r in result['segment_mappings']))

    def test_two_inputs_can_cover_one_stroke_without_inventing_ink(self):
        result = match_segments('一', [[(0,.5),(.45,.5)],[(.55,.5),(1,.5)]], ONE)
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(labels(result,1),{1})
        self.assertEqual(labels(result,2),{1})
        self.assertLess(result['target_coverage'][0],.95)
        for run in result['segment_mappings']:
            self.assertIn(run['source_stroke_number'],(1,2))

    def test_yong_inner_horizontals_joined(self):
        source=[YONG[0],YONG[1],YONG[2]+YONG[3],YONG[4]]
        result=match_segments('用',source,YONG)
        self.assertEqual(labels(result,3),{3,4})
        self.assertEqual(labels(result,4),{5})
        self.assertEqual(result['status'],'accepted',result['target_coverage'])

    def test_yong_reordered_and_reversed(self):
        order=[4,2,0,3,1]
        result=match_segments('用',[list(reversed(YONG[i])) for i in order],YONG)
        self.assertEqual(result['status'],'accepted',result['target_coverage'])
        for i,j in enumerate(order,1):
            self.assertEqual(labels(result,i),{j+1})

    def test_crossing_is_not_covering(self):
        result=match_segments('一',[[(.5,0),(.5,1)]],ONE)
        self.assertEqual(result['status'],'pending')
        self.assertEqual(result['target_coverage'],[0.0])

    def test_coverage_is_union_not_min_to_max_span(self):
        self.assertAlmostEqual(_union_length([(0,.2),(.8,1)]),.4)
        self.assertAlmostEqual(_union_length([(0,.3),(.1,.4)]),.4)

    def test_original_location_survives_anisotropic_scaling(self):
        # Three equal-original-arc samples; affine scaling makes the first edge
        # twice as long. Halfway through aligned arc is 3/8 through original arc.
        points,fractions=_sample_aligned([(0,0),(2,0),(2,1)],3)
        self.assertEqual(points[1],(1.5,0.0))
        self.assertAlmostEqual(fractions[1],.375)
        location=_source_location([(0,0),(1,0),(1,1)],fractions[1])
        self.assertEqual(location,dict(point_index=0,interpolation=.75))

    def test_invalid_paths_remain_pending(self):
        for source in ([[[0,0]]],[[[0,0],[0,0]]],[[[0,0],[float('nan'),1]]]):
            self.assertEqual(match_segments('一',source,ONE)['status'],'pending')


if __name__ == '__main__':
    unittest.main()
