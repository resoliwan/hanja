"""Whole-glyph correspondence, separate from handwriting quality.

Known SCUT labels are assumed. This is not an open-set character recognizer.
Writing order is retained in the output, but never used to choose an assignment.
"""

from dataclasses import dataclass
from itertools import combinations, permutations
import math
from statistics import median

from .geometry import dtw_distance, normalize_strokes, path_length, resample


SUPPORTED_CHARACTERS = '一二三用'


@dataclass(frozen=True)
class MappingConfig:
    points_per_stroke: int = 32
    min_vertical_separation: float = 0.03
    min_assignment_margin: float = 0.01
    tilt_probe_degrees: float = 5.0
    quality_pair_cost: float = 0.24


def _middle(points, trim):
    cut = int(len(points)*trim)
    return points[cut:len(points)-cut] if cut else points


def _center(points):
    return tuple(median(p[k] for p in points) for k in (0, 1))


def _axial_residual(angle):
    return (angle + math.pi/4) % (math.pi/2) - math.pi/4


def _tilt(strokes, trim):
    # Each stroke has equal influence. Quarter-path chords suppress point jitter
    # and short terminal hooks without deleting them from the source data.
    votes = []
    for stroke in strokes:
        points = _middle(stroke, trim)
        step = max(1, len(points)//4)
        angles = [_axial_residual(math.atan2(b[1]-a[1], b[0]-a[0]))
                  for a, b in zip(points, points[step:]) if math.dist(a, b) > 0.015]
        if angles:
            votes.append(median(angles))
    return median(votes) if votes else 0.0


def _view(strokes, count, trim, angle_offset=0):
    sampled = [resample(s, count) for s in strokes]
    angle = _tilt(sampled, trim) + math.radians(angle_offset)
    c, s = math.cos(angle), math.sin(angle)
    rotated = [[((x-.5)*c + (y-.5)*s + .5, -(x-.5)*s + (y-.5)*c + .5)
                for x, y in stroke] for stroke in sampled]
    return [_middle(stroke, trim) for stroke in rotated], math.degrees(angle)


def _descriptor(stroke):
    center = _center(stroke)
    bounds = tuple(max(p[k] for p in stroke)-min(p[k] for p in stroke) for k in (0, 1))
    travel = [sum(abs(b[k]-a[k]) for a, b in zip(stroke, stroke[1:])) for k in (0, 1)]
    total = sum(travel) or 1
    return center, bounds, tuple(v/total for v in travel)


def _pair_features(source, target):
    a, extent_a, axes_a = _descriptor(source)
    b, extent_b, axes_b = _descriptor(target)
    ca = [tuple(p[k]-a[k] for k in (0, 1)) for p in source]
    cb = [tuple(p[k]-b[k] for k in (0, 1)) for p in target]
    forward, reverse = dtw_distance(ca, cb), dtw_distance(list(reversed(ca)), cb)
    features = dict(position=math.dist(a, b), extent=math.dist(extent_a, extent_b),
                    axis_travel=abs(axes_a[0]-axes_b[0]), shape=min(forward, reverse),
                    reversed=reverse < forward)
    features['cost'] = (0.40*features['position'] + 0.20*features['extent']
                        + 0.25*features['axis_travel'] + 0.15*features['shape'])
    return features


def _layout_cost(source, target, assignment):
    costs = []
    for i, j in combinations(range(len(source)), 2):
        a, b = _center(source[i]), _center(source[j])
        c, d = _center(target[assignment[i]]), _center(target[assignment[j]])
        costs.append(math.dist((b[0]-a[0], b[1]-a[1]), (d[0]-c[0], d[1]-c[1])))
    return sum(costs)/len(costs) if costs else 0.0


def _solve(character, source, target, config):
    if character in '一二三':
        source_order = sorted(range(len(source)), key=lambda i: _center(source[i])[1])
        target_order = sorted(range(len(target)), key=lambda i: _center(target[i])[1])
        assignment = [0]*len(source)
        for i, j in zip(source_order, target_order):
            assignment[i] = j
        gaps = [_center(source[b])[1]-_center(source[a])[1] for a, b in zip(source_order, source_order[1:])]
        ambiguous = bool(gaps and min(gaps) < config.min_vertical_separation)
        return dict(assignment=tuple(assignment), ambiguous=ambiguous,
                    minimum_vertical_gap=min(gaps) if gaps else None,
                    best_cost=None, runner_up_cost=None, assignment_margin=None)
    # 用: exact search over 120 assignments, with stroke and pair-relation costs.
    pair_costs = [[_pair_features(a, b)['cost'] for b in target] for a in source]
    ranked = []
    for assignment in permutations(range(len(source))):
        appearance = sum(pair_costs[i][j] for i, j in enumerate(assignment))/len(source)
        ranked.append((0.7*appearance + 0.3*_layout_cost(source, target, assignment), assignment))
    ranked.sort()
    best, assignment = ranked[0]
    second = ranked[1][0]
    return dict(assignment=assignment, ambiguous=second-best < config.min_assignment_margin,
                minimum_vertical_gap=None, best_cost=best, runner_up_cost=second,
                assignment_margin=second-best)


def _quality_flags(character, source, target, features, assignment, tilt, config):
    flags = []
    if abs(tilt) > 10:
        flags.append('slanted_glyph')
    if any(features[i][j]['cost'] > config.quality_pair_cost for i, j in enumerate(assignment)):
        flags.append('large_template_difference')
    # 用 legitimately contains a bent stroke. Flag only extra bending relative
    # to the matched template, not the mere presence of a corner.
    def efficiency(stroke):
        return math.dist(stroke[0], stroke[-1])/path_length(stroke)
    if any(efficiency(target[j])-efficiency(source[i]) > 0.20 for i, j in enumerate(assignment)):
        flags.append('extra_bend_or_retrace')
    if character in '一二三':
        for stroke in source:
            _, extent, _ = _descriptor(stroke)
            if extent[1] > 0.55*extent[0]:
                flags.append('nonhorizontal_or_strongly_curved')
                break
    return flags


def match_strokes(character, source, target, config=MappingConfig()):
    expected = {'一': 1, '二': 2, '三': 3, '用': 5}
    if character not in expected or len(target) != expected[character]:
        raise ValueError('Only matching 一/二/三/用 templates are supported')
    result = dict(status='pending', reasons=[], mappings=[], proposed_mappings=[],
                  correspondence_basis=None, quality_flags=[], training_usage='review_required',
                  best_cost=None, runner_up_cost=None, assignment_margin=None,
                  stability=None, probes=[], pair_features=[], source_normalized=[], target_normalized=[])
    target = normalize_strokes(target, 64)
    result['target_normalized'] = target
    try:
        source = normalize_strokes(source, 64)
    except ValueError as error:
        result['reasons'] = [str(error)]
        return result
    result['source_normalized'] = source
    if len(source) != len(target):
        result['reasons'] = ['stroke_count_mismatch']
        return result
    variations = [('baseline', config.points_per_stroke, .10, 0),
                  ('all_endpoints', config.points_per_stroke, 0, 0),
                  ('central_60_percent', config.points_per_stroke, .20, 0),
                  ('sample_24', 24, .10, 0), ('sample_48', 48, .10, 0),
                  ('tilt_minus', config.points_per_stroke, .10, -config.tilt_probe_degrees),
                  ('tilt_plus', config.points_per_stroke, .10, config.tilt_probe_degrees)]
    solved = []
    for name, count, trim, angle in variations:
        a, tilt = _view(source, count, trim, angle)
        b, _ = _view(target, count, trim)
        solution = _solve(character, a, b, config)
        solution.update(name=name, source_tilt_degrees=tilt)
        solved.append(solution)
    baseline = solved[0]
    assignment = baseline['assignment']
    a, tilt = _view(source, config.points_per_stroke, .10)
    b, _ = _view(target, config.points_per_stroke, .10)
    features = [[_pair_features(s, t) for t in b] for s in a]
    agreement = sum(s['assignment'] == assignment for s in solved)
    reasons = []
    if any(s['ambiguous'] for s in solved):
        reasons.append('ambiguous_structure')
    if agreement != len(solved):
        reasons.append('unstable_correspondence')
    proposed = [dict(source_stroke_number=i+1, svg_stroke_number=j+1,
                     cost=features[i][j]['cost'], reversed=features[i][j]['reversed'])
                for i, j in enumerate(assignment)]
    flags = _quality_flags(character, source, target, features, assignment, tilt, config)
    result.update(
        status='accepted' if not reasons else 'pending', reasons=reasons,
        mappings=proposed if not reasons else [], proposed_mappings=proposed,
        correspondence_basis='unique_given_label' if character == '一' else 'deskewed_vertical_order' if character in '二三' else 'whole_glyph_structure',
        quality_flags=flags, training_usage='candidate' if not reasons and not flags else 'review_required',
        stability=agreement/len(solved), probes=solved, pair_features=features,
        best_cost=baseline['best_cost'], runner_up_cost=baseline['runner_up_cost'],
        assignment_margin=baseline['assignment_margin'],
    )
    return result
