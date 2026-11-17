"""Direction-consistent blocks, weak stroke-order hints and reversible anchoring.

Geometry can override order. An anchor is selected using whole-glyph evidence,
never its first-stroke fit alone. Both hypotheses remain in the audit output.
"""

from dataclasses import dataclass
import math

import numpy as np

from .geometry import normalize_strokes, path_length
from .segment_matcher import (
    SegmentConfig, SUPPORTED_CHARACTERS, _align, _match_aligned,
    _union_length, match_segments,
)


@dataclass(frozen=True)
class GuidedConfig(SegmentConfig):
    use_order_prior: bool = True
    use_anchor_alignment: bool = True
    simplify_tolerance: float = .025
    corner_angle_degrees: float = 40.0
    minimum_block_length: float = .06
    order_geometry_window: float = .04
    order_prior_weight: float = .035
    minimum_block_support: float = .35
    max_anchor_shift: float = .50
    anchor_score_tolerance: float = .005


def _simplify_indices(points, tolerance):
    """RDP candidates only; not every vertex becomes a label boundary."""
    if len(points) <= 2:
        return [0,len(points)-1]
    a,b=np.array(points[0]),np.array(points[-1])
    delta=b-a
    if np.dot(delta,delta) < 1e-12:
        distances=np.linalg.norm(np.array(points)-a,axis=1)
    else:
        t=np.clip(((np.array(points)-a)*delta).sum(1)/np.dot(delta,delta),0,1)
        distances=np.linalg.norm(np.array(points)-(a+t[:,None]*delta),axis=1)
    index=int(np.argmax(distances))
    if distances[index] <= tolerance or index in (0,len(points)-1):
        return [0,len(points)-1]
    left=_simplify_indices(points[:index+1],tolerance)
    right=_simplify_indices(points[index:],tolerance)
    return left[:-1]+[index+i for i in right]


def _blocks(points, config):
    indices=_simplify_indices(points,config.simplify_tolerance)
    boundaries=[0]
    for a,b,c in zip(indices,indices[1:],indices[2:]):
        u,v=np.array(points[b])-points[a],np.array(points[c])-points[b]
        cosine=np.dot(u,v)/max(1e-12,np.linalg.norm(u)*np.linalg.norm(v))
        angle=math.degrees(math.acos(float(np.clip(cosine,-1,1))))
        if angle >= config.corner_angle_degrees and path_length(points[boundaries[-1]:b+1]) >= config.minimum_block_length:
            boundaries.append(b)
    boundaries.append(len(points)-1)
    return list(zip(boundaries,boundaries[1:]))


class _BlockSelector:
    def __init__(self, target_count, config):
        self.config=config
        self.coverage=[[] for _ in range(target_count)]
        self.last=None
        self.order_choices=0

    def _order_cost(self, number):
        unfinished=[i+1 for i,c in enumerate(self.coverage) if _union_length(c)<self.config.min_target_coverage]
        if unfinished and number==unfinished[0]:
            return 0.0
        if number==self.last:
            return .25
        if self.last is None:
            return 1.0
        return .5 if number>self.last else 1.0

    def __call__(self, points, template, best_indices, best_costs, distances, cosines, projection):
        config=self.config
        _,_,lengths,numbers,offsets,totals=template
        ink_lengths=np.linalg.norm(np.diff(np.array(points),axis=0),axis=1)
        chosen=best_indices[np.arange(len(points)-1),np.argmin(best_costs,axis=1)].copy()
        labels=[None]*(len(points)-1)
        reasons=['unsupported_direction_block']*len(labels)
        for start,end in _blocks(points,config):
            candidates=[]
            for column in range(best_indices.shape[1]):
                rows=np.arange(start,end); js=best_indices[rows,column]
                valid=(distances[rows,js]<=config.max_distance)&(cosines[rows,js]>=config.min_direction_cosine)
                support=float(np.sum(ink_lengths[rows]*valid)/max(1e-12,np.sum(ink_lengths[rows])))
                if support<config.minimum_block_support:
                    continue
                # All points in the direction-consistent block vote together.
                cost=float(np.average(np.minimum(best_costs[start:end,column],.5),weights=ink_lengths[start:end]))
                candidates.append((cost,column+1))
            if not candidates:
                continue
            candidates.sort()
            best=candidates[0][0]
            close=[c for c in candidates if c[0]<=best+config.order_geometry_window]
            if config.use_order_prior:
                ranked=sorted((cost+config.order_prior_weight*self._order_cost(number),cost,number) for cost,number in close)
            else:
                ranked=sorted((cost,cost,number) for cost,number in close)
            if len(ranked)>1 and ranked[1][0]-ranked[0][0]<config.min_candidate_margin:
                for i in range(start,end): reasons[i]='ambiguous_direction_block'
                continue
            number=ranked[0][2]
            if number!=candidates[0][1]: self.order_choices+=1
            any_matched=False
            for i in range(start,end):
                j=int(best_indices[i,number-1]); chosen[i]=j
                if distances[i,j]>config.max_distance:
                    reasons[i]='far_from_template'
                elif cosines[i,j]<config.min_direction_cosine:
                    reasons[i]='direction_mismatch'
                else:
                    labels[i],reasons[i]=number,None
                    center=(offsets[j]+projection[i,j]*lengths[j])/totals[j]
                    half=ink_lengths[i]*cosines[i,j]/(2*totals[j])
                    self.coverage[number-1].append((max(0.,center-half),min(1.,center+half)))
                    any_matched=True
            if any_matched: self.last=number
        return labels,reasons,chosen


def _anchor_shift(source, target, config):
    def first_stable(points):
        for a,b in _blocks(points,config):
            if path_length(points[a:b+1])>=config.minimum_block_length:
                return points[a:b+1]
        return None
    a,b=first_stable(source[0]),first_stable(target[0])
    if a is None or b is None:
        return None,'no_stable_initial_block'
    u,v=np.array(a[-1])-a[0],np.array(b[-1])-b[0]
    cosine=abs(float(np.dot(u,v)/max(1e-12,np.linalg.norm(u)*np.linalg.norm(v))))
    if cosine<config.min_direction_cosine:
        return None,'first_block_direction_disagrees'
    # Match early robust locations, not length or scale. The WHOLE glyph fixes
    # scale already, so a short first stroke cannot enlarge the entire glyph.
    if np.dot(u,v)<0:
        a=list(reversed(a))
    anchor_a=np.median(np.array(a[:max(2,len(a)//5)]),axis=0)
    anchor_b=np.median(np.array(b[:max(2,len(b)//5)]),axis=0)
    shift=anchor_b-anchor_a
    if np.linalg.norm(shift)>config.max_anchor_shift:
        return None,'first_block_requires_large_shift'
    return shift,None


def _fit_score(result):
    covered=result['target_coverage']
    mean=sum(covered)/len(covered)
    # Same global criterion for both hypotheses; first-stroke fit does not
    # appear here. No reviewed labels enter alignment selection.
    return (result['unmatched_fraction'] + .5*(1-mean)
            + .5*(1-min(covered)))


def match_guided(character, raw_source, raw_target, config=GuidedConfig()):
    expected={'一':1,'二':2,'三':3,'四':5,'用':5}
    if character not in expected or len(raw_target)!=expected[character]:
        raise ValueError('Unsupported character or template stroke count')
    try:
        source=normalize_strokes(raw_source,192)
    except ValueError:
        return match_segments(character,raw_source,raw_target,config)
    target=normalize_strokes(raw_target,96)
    source,target,transform=_align(source,target)
    selector=_BlockSelector(len(target),config)
    central=_match_aligned(raw_source,source,target,dict(transform,mode='center'),config,selector)
    central['order_prior_overrides']=selector.order_choices
    central_score=_fit_score(central)
    hypotheses=[dict(mode='center',score=central_score,status=central['status'],
                     coverage=central['target_coverage'],unmatched=central['unmatched_fraction'])]
    selected=central; anchor_reason='disabled'
    if config.use_anchor_alignment:
        shift,anchor_reason=_anchor_shift(source,target,config)
        if shift is not None:
            anchored=[[tuple(np.array(p)+shift) for p in stroke] for stroke in source]
            selector=_BlockSelector(len(target),config)
            first=_match_aligned(raw_source,anchored,target,
                                 dict(transform,mode='first_stroke',shift_xy=shift.tolist()),config,selector)
            first['order_prior_overrides']=selector.order_choices
            score=_fit_score(first)
            hypotheses.append(dict(mode='first_stroke',score=score,status=first['status'],
                                   coverage=first['target_coverage'],unmatched=first['unmatched_fraction']))
            if score<=central_score+config.anchor_score_tolerance:
                selected=first
    selected['alignment_hypotheses']=hypotheses
    selected['anchor_rejection_reason']=anchor_reason
    selected['center_fallback']=bool(config.use_anchor_alignment and selected is central)
    selected['correspondence_basis']='direction_blocks_with_weak_order_and_reversible_anchor'
    return selected
