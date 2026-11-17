"""Local segment-to-template projection, allowing many-to-many correspondence.

No stroke-order prior, assignment search, or invented ink across pen lifts.
Unmapped intervals remain explicit. Coverage is an interval UNION, not a span.
"""

from dataclasses import dataclass
import math
from statistics import median

import numpy as np

from .geometry import normalize_strokes, path_length, resample


SUPPORTED_CHARACTERS = '一二三四用'


@dataclass(frozen=True)
class SegmentConfig:
    segment_length: float = 0.025
    max_distance: float = 0.16
    min_direction_cosine: float = 0.65
    direction_weight: float = 0.12
    min_candidate_margin: float = 0.008
    min_run_length: float = 0.05
    max_ambiguous_bridge: float = 0.05
    min_target_coverage: float = 0.45
    max_unmapped_fraction: float = 0.45


def _tilt(strokes):
    votes = []
    for stroke in strokes:
        points = resample(stroke, 48)[5:-5]
        angles = []
        for a, b in zip(points, points[4:]):
            if math.dist(a, b) < .015:
                continue
            angle = math.atan2(b[1]-a[1], b[0]-a[0])
            angles.append((angle+math.pi/4) % (math.pi/2)-math.pi/4)
        if angles:
            votes.append(median(angles))
    return median(votes) if votes else 0.0


def _rotate(strokes, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [[((x-.5)*c+(y-.5)*s+.5, -(x-.5)*s+(y-.5)*c+.5) for x,y in stroke] for stroke in strokes]


def _align(source, target):
    source_angle, target_angle = _tilt(source), _tilt(target)
    a, b = _rotate(source, source_angle), _rotate(target, target_angle)
    pa, pb = np.array([p for s in a for p in s]), np.array([p for s in b for p in s])
    low_a, high_a = pa.min(0), pa.max(0)
    low_b, high_b = pb.min(0), pb.max(0)
    extent_a, extent_b = high_a-low_a, high_b-low_b
    # Whole-character aspect fitting tolerates compressed handwriting. Do not
    # magnify tiny vertical variation of 一 into the SVG terminal brush shape.
    scale = np.full(2, float(extent_b.max()/extent_a.max()))
    if extent_a.min() > .10 and extent_b.min() > .10:
        scale = np.clip(extent_b/extent_a, .4, 2.5)
    center_a, center_b = (low_a+high_a)/2, (low_b+high_b)/2
    aligned = [[tuple((np.array(p)-center_a)*scale+center_b) for p in s] for s in a]
    return aligned, b, dict(source_rotation_degrees=math.degrees(source_angle),
                            template_rotation_degrees=math.degrees(target_angle),
                            scale_xy=scale.tolist())


def _template_segments(target):
    starts, vectors, lengths, numbers, offsets, totals = [], [], [], [], [], []
    for number, stroke in enumerate(target, 1):
        length, offset = path_length(stroke), 0.0
        for a, b in zip(stroke, stroke[1:]):
            segment_length = math.dist(a, b)
            if segment_length <= 1e-9:
                continue
            starts.append(a); vectors.append((b[0]-a[0], b[1]-a[1]))
            lengths.append(segment_length); numbers.append(number)
            offsets.append(offset); totals.append(length)
            offset += segment_length
    return [np.asarray(v) for v in (starts, vectors, lengths, numbers, offsets, totals)]


def _runs(labels):
    start = 0
    for index in range(1, len(labels)+1):
        if index == len(labels) or labels[index] != labels[start]:
            yield start, index, labels[start]
            start = index


def _union_length(intervals):
    covered, end = 0.0, 0.0
    for left, right in sorted(intervals):
        covered += max(0.0, right-max(end, left))
        end = max(end, right)
    return min(1.0, covered)


def _source_location(raw, fraction):
    """Locate an arc fraction in the ORIGINAL coordinate list (0-based index)."""
    distance, elapsed = fraction*path_length(raw), 0.0
    for index, (a, b) in enumerate(zip(raw, raw[1:])):
        length = math.dist(a, b)
        if elapsed+length >= distance and length > 0:
            return dict(point_index=index, interpolation=max(0., min(1., (distance-elapsed)/length)))
        elapsed += length
    return dict(point_index=len(raw)-2, interpolation=1.0)


def _sample_aligned(points, count):
    """Keep original arc fractions through anisotropic whole-glyph alignment.

    The incoming 192 points were sampled uniformly in ORIGINAL arc length.
    Aligned arc length is different, so track the originating edge and fraction.
    """
    cumulative = [0.0]
    for a,b in zip(points,points[1:]):
        cumulative.append(cumulative[-1]+math.dist(a,b))
    sampled, fractions, edge = [], [], 1
    for i in range(count):
        distance = cumulative[-1]*i/(count-1)
        while edge < len(points)-1 and cumulative[edge] < distance:
            edge += 1
        span = cumulative[edge]-cumulative[edge-1]
        t = (distance-cumulative[edge-1])/span if span else 0
        sampled.append(tuple(points[edge-1][k]*(1-t)+points[edge][k]*t for k in (0,1)))
        fractions.append((edge-1+t)/(len(points)-1))
    return sampled, fractions


def _match_path(points, template, target_count, config, selector=None):
    starts, vectors, lengths, numbers, offsets, totals = template
    a, b = np.asarray(points[:-1]), np.asarray(points[1:])
    delta = b-a
    ink_lengths = np.linalg.norm(delta, axis=1)
    middle = (a+b)/2
    projection = np.clip(((middle[:,None,:]-starts)*vectors).sum(2)/(lengths**2), 0, 1)
    distances = np.linalg.norm(middle[:,None,:]-(starts+projection[:,:,None]*vectors), axis=2)
    cosines = np.abs((delta[:,None,:]*vectors).sum(2)/np.maximum(ink_lengths[:,None]*lengths, 1e-12))
    costs = distances + config.direction_weight*(1-cosines)
    best_indices = np.column_stack([np.where(numbers == n)[0][np.argmin(costs[:, numbers==n], axis=1)]
                                    for n in range(1, target_count+1)])
    best_costs = np.take_along_axis(costs, best_indices, axis=1)
    if selector is not None:
        labels, reasons, chosen = selector(points, template, best_indices, best_costs, distances, cosines, projection)
    else:
        rank = np.argsort(best_costs, axis=1, kind='stable')
        chosen = best_indices[np.arange(len(a)),rank[:,0]]
        gap = best_costs[np.arange(len(a)),rank[:,1]]-best_costs[np.arange(len(a)),rank[:,0]] if target_count > 1 else np.full(len(a), np.inf)
        labels, reasons = [], []
        for i, j in enumerate(chosen):
            reason = ('far_from_template' if distances[i,j] > config.max_distance else
                      'direction_mismatch' if cosines[i,j] < config.min_direction_cosine else
                      'ambiguous_crossing' if gap[i] < config.min_candidate_margin else None)
            labels.append(int(numbers[j]) if reason is None else None)
            reasons.append(reason)
    # Only bridge short ambiguous crossings bracketed by the SAME label. Never
    # fill distant/direction-mismatched ink or bridge between separate inputs.
    for start, end, label in list(_runs(labels)):
        if (label is None and start > 0 and end < len(labels) and labels[start-1] is not None
                and labels[start-1] == labels[end] and sum(ink_lengths[start:end]) <= config.max_ambiguous_bridge
                and all(reason == 'ambiguous_crossing' for reason in reasons[start:end])):
            for i in range(start, end):
                labels[i] = labels[start-1]
                chosen[i] = best_indices[i, labels[i]-1]
                reasons[i] = None
    for start, end, label in list(_runs(labels)):
        if label is not None and sum(ink_lengths[start:end]) < config.min_run_length:
            for i in range(start, end):
                labels[i], reasons[i] = None, 'insufficient_contiguous_support'
    intervals = []
    for i, j in enumerate(chosen):
        if labels[i] is None:
            intervals.append(None)
            continue
        center = (offsets[j]+projection[i,j]*lengths[j])/totals[j]
        half_span = ink_lengths[i]*cosines[i,j]/(2*totals[j])
        left, right = max(0.,center-half_span), min(1.,center+half_span)
        intervals.append((float(left),float(right)))
    return labels, reasons, intervals, ink_lengths.tolist()


def match_segments(character, raw_source, raw_target, config=SegmentConfig()):
    expected = {'一':1, '二':2, '三':3, '四':5, '用':5}
    if character not in expected or len(raw_target) != expected[character]:
        raise ValueError('Unsupported character or template stroke count')
    result = dict(status='pending', reasons=[], mappings=[], proposed_mappings=[], segment_mappings=[],
                  source_normalized=[], target_normalized=[], target_coverage=[],
                  training_usage='review_required', quality_flags=[], best_cost=None, runner_up_cost=None,
                  assignment_margin=None, stability=None, unmatched_fraction=1.0)
    target = normalize_strokes(raw_target, 96)
    result['target_normalized'] = target
    try:
        # Same number of points is not required between input strokes.
        source = normalize_strokes(raw_source, 192)
    except ValueError as error:
        result['reasons'] = [str(error)]
        return result
    source, target, transform = _align(source, target)
    result['alignment'] = transform
    result['target_normalized'] = target
    return _match_aligned(raw_source, source, target, transform, config)


def _match_aligned(raw_source, source, target, transform, config, selector=None):
    """Shared interval assembly for a single fixed alignment hypothesis."""
    result = dict(status='pending', reasons=[], mappings=[], proposed_mappings=[], segment_mappings=[],
                  source_normalized=[], target_normalized=target, target_coverage=[], alignment=transform,
                  training_usage='review_required', quality_flags=[], best_cost=None, runner_up_cost=None,
                  assignment_margin=None, stability=None, unmatched_fraction=1.0)
    prepared = [_sample_aligned(s, max(2, math.ceil(path_length(s)/config.segment_length)+1)) for s in source]
    sampled = [s for s,_ in prepared]
    result['source_normalized'] = sampled
    template = _template_segments(target)
    coverage = [[] for _ in target]
    runs, total_ink, unmapped = [], 0.0, 0.0
    for number, points in enumerate(sampled, 1):
        labels, reasons, intervals, lengths = _match_path(points, template, len(target), config, selector)
        cumulative = [0.0]
        for length in lengths:
            cumulative.append(cumulative[-1]+length)
        total_ink += cumulative[-1]
        for i, label in enumerate(labels):
            if label is None:
                unmapped += lengths[i]
            elif intervals[i] is not None:
                coverage[label-1].append(intervals[i])
        for start, end, label in _runs(labels):
            original_fractions = prepared[number-1][1]
            fraction_start, fraction_end = original_fractions[start], original_fractions[end]
            connector = (label is None and start>0 and end<len(labels) and labels[start-1] is not None
                         and labels[end] is not None and labels[start-1] != labels[end])
            runs.append(dict(source_stroke_number=number, svg_stroke_number=label,
                             sample_start=start, sample_end=end,
                             source_fraction_start=fraction_start, source_fraction_end=fraction_end,
                             original_start=_source_location(raw_source[number-1], fraction_start),
                             original_end=_source_location(raw_source[number-1], fraction_end),
                             kind='matched' if label else 'connector_candidate' if connector else 'unmapped',
                             reasons=sorted({r for r in reasons[start:end] if r}),
                             reference_intervals=[intervals[i] for i in range(start,end) if intervals[i] is not None]))
    covered = [_union_length(intervals) for intervals in coverage]
    reasons = []
    if any(value < config.min_target_coverage for value in covered):
        reasons.append('incomplete_target_coverage')
    if unmapped/max(total_ink, 1e-9) > config.max_unmapped_fraction:
        reasons.append('excess_unmapped_ink')
    proposed = []
    for number in range(1, len(source)+1):
        for target_number in sorted({r['svg_stroke_number'] for r in runs if r['source_stroke_number']==number and r['svg_stroke_number'] is not None}):
            proposed.append(dict(source_stroke_number=number, svg_stroke_number=target_number))
    result.update(status='pending' if reasons else 'accepted', reasons=reasons,
                  segment_mappings=runs, proposed_mappings=proposed,
                  mappings=proposed if not reasons else [],
                  target_coverage=covered, unmatched_fraction=unmapped/max(total_ink,1e-9),
                  # Local projection can pass coverage while mislabeling a
                  # connector. These pilot answers require review before training.
                  training_usage='review_required')
    return result
