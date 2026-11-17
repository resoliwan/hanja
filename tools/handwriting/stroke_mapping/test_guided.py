"""Order is a tie-breaker; first-stroke anchoring must be reversible."""

from dataclasses import replace
import math
import unittest

from tools.handwriting.stroke_mapping.guided_matcher import GuidedConfig, match_guided, _blocks
from tools.handwriting.stroke_mapping.test_segments import ONE, TWO, YONG, labels


class GuidedMappingTests(unittest.TestCase):
    def test_nonstandard_yong_order_overrides_prior(self):
        order=[0,1,4,2,3]
        result=match_guided('用',[YONG[i] for i in order],YONG)
        self.assertEqual(result['status'],'accepted')
        for i,j in enumerate(order,1): self.assertEqual(labels(result,i),{j+1})

    def test_wrong_first_stroke_hypothesis_falls_back_globally(self):
        template=[[(.1,.4),(.9,.4)],[(0,.6),(1,.6)]]
        result=match_guided('二',list(reversed(template)),template)
        self.assertEqual(len(result['alignment_hypotheses']),2)
        self.assertEqual(result['alignment']['mode'],'center')
        self.assertTrue(result['center_fallback'])
        self.assertEqual(labels(result,1),{2})
        self.assertEqual(labels(result,2),{1})

    def test_anchor_preserves_whole_glyph_scale(self):
        result=match_guided('二',TWO,TWO)
        central=match_guided('二',TWO,TWO,replace(GuidedConfig(),use_anchor_alignment=False))
        self.assertEqual(result['alignment']['scale_xy'],central['alignment']['scale_xy'])
        self.assertEqual(result['alignment']['mode'],'first_stroke')

    def test_long_straight_block_does_not_split_for_small_jitter(self):
        points=[(i/50,.5+.004*math.sin(i)) for i in range(51)]
        self.assertEqual(_blocks(points,GuidedConfig()),[(0,50)])

    def test_joined_horizontals_remain_many_to_many(self):
        source=[YONG[0],YONG[1],YONG[2]+YONG[3],YONG[4]]
        result=match_guided('用',source,YONG)
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(labels(result,3),{3,4})
        self.assertEqual(labels(result,4),{5})

    def test_direction_mismatch_is_not_overridden_by_order(self):
        result=match_guided('一',[[(.5,0),(.5,1)]],ONE)
        self.assertEqual(result['status'],'pending')
        self.assertEqual(result['target_coverage'],[0.0])

    def test_order_can_resolve_close_candidates_without_forcing_completion(self):
        target=[[(0,.45),(1,.45)],[(0,.55),(1,.55)]]
        source=[[(0,.5),(1,.5)]]
        config=replace(GuidedConfig(),use_anchor_alignment=False)
        hinted=match_guided('二',source,target,config)
        unhinted=match_guided('二',source,target,replace(config,use_order_prior=False))
        self.assertEqual(labels(hinted,1),{1})
        self.assertEqual(labels(unhinted,1),set())
        self.assertEqual(hinted['status'],'pending')


if __name__=='__main__':
    unittest.main()
