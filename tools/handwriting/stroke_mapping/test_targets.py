"""Target bundle ownership, pen lifts, joined paths and directed matching."""

from dataclasses import replace
import unittest

from tools.handwriting.stroke_mapping.target_matcher import TargetMappingConfig, match_targets
from tools.handwriting.stroke_mapping.test_segments import ONE, TWO, YONG, labels


class TargetMappingTests(unittest.TestCase):
    def test_nonadjacent_slightly_different_repeat_maps_to_same_target(self):
        duplicate = [(x+.008,y-.005) for x,y in TWO[0]]
        result = match_targets('二', TWO+[duplicate], TWO)
        self.assertEqual([labels(result,i) for i in (1,2,3)], [{1},{2},{1}])
        self.assertEqual(result['target_assignment'][:2], [1,1])

    def test_repeated_bent_stroke_preserves_other_assignments(self):
        duplicate = [(x-.007,y+.004) for x,y in YONG[1]]
        result = match_targets('用', YONG+[duplicate], YONG)
        self.assertEqual([labels(result,i) for i in range(1,7)], [{1},{2},{3},{4},{5},{2}])

    def test_multiple_repetitions_can_share_one_target(self):
        source = TWO+[[(x+.004,y) for x,y in TWO[0]],[(x-.004,y+.003) for x,y in TWO[0]]]
        result = match_targets('二', source, TWO)
        self.assertEqual([labels(result,i) for i in range(1,5)], [{1},{2},{1},{1}])

    def test_close_distinct_references_are_not_collapsed(self):
        target = [[(0,.49),(1,.49)],[(0,.51),(1,.51)]]
        result = match_targets('二', target, target)
        self.assertEqual([labels(result,i) for i in (1,2)], [{1},{2}])

    def test_two_inputs_belong_to_one_reference(self):
        result = match_targets('一', [[(0,.5),(.45,.5)],[(.55,.5),(1,.5)]], ONE)
        self.assertEqual(result['status'], 'mapped')
        self.assertEqual(labels(result, 1), {1})
        self.assertEqual(labels(result, 2), {1})
        self.assertLess(result['target_coverage'][0], .95)
        self.assertEqual(len(result['target_groups'][0]['source_intervals']), 2)

    def test_three_inputs_can_form_one_reference(self):
        source = [[(0,.5),(.3,.5)],[(.35,.5),(.65,.5)],[(.7,.5),(1,.5)]]
        result = match_targets('一', source, ONE)
        self.assertEqual([labels(result, i) for i in (1,2,3)], [{1}]*3)

    def test_bent_reference_can_cross_a_pen_lift(self):
        source = [YONG[0], YONG[1][:2], YONG[1][1:], *YONG[2:]]
        result = match_targets('用', source, YONG)
        self.assertEqual([labels(result, i) for i in range(1,7)], [{1},{2},{2},{3},{4},{5}])

    def test_joined_strokes_leave_connector_unassigned(self):
        result = match_targets('二', [TWO[0]+TWO[1]], TWO)
        self.assertEqual(labels(result, 1), {1,2})
        self.assertTrue(any(r['kind']=='connector_candidate' for r in result['segment_mappings']))
        runs = result['segment_mappings']
        for a,b in zip(runs, runs[1:]):
            self.assertEqual(a['source_fraction_end'], b['source_fraction_start'])

    def test_joined_yong_horizontals(self):
        source = [YONG[0],YONG[1],YONG[2]+YONG[3],YONG[4]]
        result = match_targets('用', source, YONG)
        self.assertEqual([labels(result,i) for i in range(1,5)], [{1},{2},{3,4},{5}])

    def test_geometry_can_override_stroke_order(self):
        order = [0,1,4,2,3]
        result = match_targets('用', [YONG[i] for i in order], YONG)
        self.assertEqual([labels(result,i) for i in range(1,6)], [{j+1} for j in order])

    def test_one_input_cannot_fill_two_reference_strokes(self):
        # Two identical target strokes compete for exactly the same real ink.
        result = match_targets('二', ONE, ONE+ONE)
        self.assertEqual(len(result['mappings']), 1)
        self.assertEqual(len(result['missing_target_strokes']), 1)
        self.assertEqual(result['status'], 'mapped')  # prediction, not truth

    def test_signed_direction_changes_matching_cost(self):
        config = replace(TargetMappingConfig(), use_anchor_alignment=False)
        forward = match_targets('一', ONE, ONE, config)
        reverse = match_targets('一', [list(reversed(ONE[0]))], ONE, config)
        self.assertLess(forward['best_cost'], reverse['best_cost'])

    def test_original_intervals_are_kept_for_split_inputs(self):
        source = [[(0,.5),(.4,.5)],[(.6,.5),(1,.5)]]
        result = match_targets('一', source, ONE)
        for run in result['segment_mappings']:
            self.assertEqual(run['original_start'], dict(point_index=0, interpolation=0.))
            self.assertEqual(run['original_end'], dict(point_index=0, interpolation=1.))

    def test_invalid_input_is_explicit_not_pending(self):
        for source in ([], [[(0,0)]], [[(0,0),(0,0)]]):
            result = match_targets('一', source, ONE)
            self.assertEqual(result['status'], 'invalid')
            self.assertEqual(result['mappings'], [])
            self.assertEqual(result['target_assignment'], [0]*33)
            self.assertEqual(result['target_similarity'], [0.]*33)

    def test_binary_and_similarity_vectors_are_fixed_width(self):
        result = match_targets('二', TWO, TWO)
        self.assertEqual(result['target_count'], 2)
        self.assertEqual(result['target_assignment'], [1,1]+[0]*31)
        self.assertEqual(len(result['target_similarity']), 33)
        self.assertEqual(result['target_similarity'][2:], [0.]*31)
        self.assertTrue(all(0<=score<=1 for score in result['target_similarity']))
        self.assertFalse(result['score_normalization']['calibrated_probability'])

    def test_poor_fit_has_lower_score_without_changing_assignment(self):
        horizontal = match_targets('一', ONE, ONE)
        vertical = match_targets('一', [[(.5,0),(.5,1)]], ONE)
        self.assertEqual(horizontal['target_assignment'], vertical['target_assignment'])
        self.assertGreater(horizontal['target_similarity'][0], vertical['target_similarity'][0])
        self.assertGreater(vertical['target_similarity'][0], 0.)
        self.assertLess(vertical['target_similarity'][0], .1)

    def test_unused_extra_ink_does_not_lower_matching_stroke_score(self):
        horizontal = match_targets('一', ONE, ONE)
        cross = match_targets('一', ONE+[[(.5,0),(.5,1)]], ONE)
        self.assertGreater(cross['unmatched_fraction'], 0.)
        self.assertEqual(horizontal['target_similarity'], cross['target_similarity'])

    def test_similarity_scale_changes_output_not_mapping(self):
        source = [list(reversed(ONE[0]))]
        small = match_targets('一', source, ONE, replace(TargetMappingConfig(), similarity_cost_scale=.01))
        large = match_targets('一', source, ONE, replace(TargetMappingConfig(), similarity_cost_scale=1.))
        self.assertEqual(small['target_assignment'], large['target_assignment'])
        self.assertEqual(small['best_cost'], large['best_cost'])
        self.assertEqual(small['target_similarity'][0], 0.)
        self.assertGreater(large['target_similarity'][0], 0.)

    def test_similarity_scale_rejects_invalid_numbers(self):
        for scale in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                TargetMappingConfig(similarity_cost_scale=scale)


if __name__ == '__main__':
    unittest.main()
