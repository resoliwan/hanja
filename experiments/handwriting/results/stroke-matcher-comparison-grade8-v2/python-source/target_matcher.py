"""Offline target-first correspondence for a known, complete handwritten glyph.

Each SVG stroke selects one bundle of consecutive or repeated source intervals. Bundles
may cross pen lifts, but never invent ink across a lift. A joint beam search
enforces exclusive ownership of every source interval. Scores are costs, not
probabilities; no acceptance/abstention threshold is used to hide predictions.
"""

from dataclasses import dataclass
import math

import numpy as np

from .geometry import normalize_strokes, path_length, resample
from .guided_matcher import _anchor_shift, _blocks
from .segment_matcher import _align, _source_location, _union_length


@dataclass(frozen=True)
class TargetMappingConfig:
    points_per_curve: int = 24
    simplify_tolerance: float = .025
    corner_angle_degrees: float = 40.0
    minimum_block_length: float = .06
    max_bundle_inputs: int = 3
    candidates_per_target: int = 24
    beam_width: int = 256
    split_penalty: float = .045
    merge_penalty: float = .015
    order_penalty: float = .025
    unused_ink_penalty: float = .40
    missing_target_cost: float = .60
    direction_weight: float = .12
    use_anchor_alignment: bool = True
    min_direction_cosine: float = .65
    max_anchor_shift: float = .50
    coverage_distance: float = .16
    similarity_cost_scale: float = .60
    duplicate_mean_distance: float = .020
    duplicate_max_distance: float = .035
    duplicate_min_length_ratio: float = .85

    def __post_init__(self):
        if not math.isfinite(self.similarity_cost_scale) or self.similarity_cost_scale <= 0:
            raise ValueError('similarity_cost_scale must be finite and positive')


@dataclass(frozen=True)
class _Atom:
    source: int
    start: int
    end: int
    length: float


@dataclass(frozen=True)
class _Bundle:
    atoms: tuple
    mask: int
    spans: tuple
    length: float
    cuts: int
    repeated: bool = False


def _target_scores(fits, config):
    """Normalize selected bundle fit, independently for each SVG stroke.

    Unassigned and padding slots are zero. Binary assignment is NOT a
    threshold on similarity. Fixed scaling keeps unrelated targets and extra
    unused ink from changing a selected bundle's score. This is not calibrated
    correctness probability or geometric coverage.
    """
    assigned = [int(fit is not None) for fit in fits]
    similarity = [max(0., min(1., 1-float(fit)/config.similarity_cost_scale))
                  if fit is not None else 0. for fit in fits]
    padding = 33-len(fits)
    return dict(target_count=len(fits), target_assignment=assigned+[0]*padding,
                target_similarity=similarity+[0.]*padding,
                score_normalization=dict(method='clipped_linear', cost_scale=config.similarity_cost_scale,
                                         calibrated_probability=False))


def _atoms(source, config):
    return [_Atom(i, a, b, path_length(stroke[a:b+1]))
            for i, stroke in enumerate(source)
            for a, b in _blocks(stroke, config)]


def _bundles(atoms, source, config):
    """Keep pen-lift boundaries separate, even when a bundle uses both sides."""
    result = []
    for first in range(len(atoms)):
        spans, length, mask = [], 0.0, 0
        for last in range(first, len(atoms)):
            atom = atoms[last]
            if atom.source - atoms[first].source >= config.max_bundle_inputs:
                break
            if spans and spans[-1][0] == atom.source:
                spans[-1] = (atom.source, spans[-1][1], atom.end)
            else:
                spans.append((atom.source, atom.start, atom.end))
            length += atom.length
            mask |= 1 << last
            cuts = int(spans[0][1] > 0) + int(spans[-1][2] < len(source[spans[-1][0]])-1)
            result.append(_Bundle(tuple(range(first, last+1)), mask, tuple(spans), length, cuts))
    # Repeated whole strokes can be separated by any number of pen inputs.
    # Add optional joint candidates; keep the individual candidates so nearby
    # legitimate strokes can still occupy separate SVG targets in the solve.
    sampled = [np.array(resample(s,32)) for s in source]
    lengths = [path_length(s) for s in source]
    duplicate_sets = set()
    for i,a in enumerate(sampled):
        peers = [i]
        for j in range(i+1,len(source)):
            if min(lengths[i],lengths[j])/max(lengths[i],lengths[j]) < config.duplicate_min_length_ratio:
                continue
            distances = min((np.linalg.norm(a-b,axis=1) for b in (sampled[j],sampled[j][::-1])),
                            key=lambda d:float(d.mean()))
            if distances.mean()<=config.duplicate_mean_distance and distances.max()<=config.duplicate_max_distance:
                peers.append(j)
                duplicate_sets.add((i,j))
        if len(peers)>2:
            duplicate_sets.add(tuple(peers))
    for members in sorted(duplicate_sets):
        indices = tuple(k for k,a in enumerate(atoms) if a.source in members)
        result.append(_Bundle(indices,sum(1<<k for k in indices),
                              tuple((i,0,len(source[i])-1) for i in members),
                              sum(lengths[i] for i in members),0,True))
    return result


def _describe(pieces, count):
    """Sample real ink arc length; direction windows never cross a pen lift."""
    starts, deltas, tangents = [], [], []
    for piece in pieces:
        points = np.array(resample(piece, max(12, count)))
        for i, (a, b) in enumerate(zip(points, points[1:])):
            delta = b-a
            if np.linalg.norm(delta) < 1e-10:
                continue
            tangent = points[min(len(points)-1, i+3)] - points[max(0, i-2)]
            tangent /= max(float(np.linalg.norm(tangent)), 1e-10)
            starts.append(a); deltas.append(delta); tangents.append(tangent)
    starts, deltas, tangents = map(np.asarray, (starts, deltas, tangents))
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    distances = (np.arange(count)+.5)/count*cumulative[-1]
    indices = np.minimum(np.searchsorted(cumulative, distances, side='right')-1, len(lengths)-1)
    fractions = (distances-cumulative[indices])/lengths[indices]
    points = starts[indices] + fractions[:, None]*deltas[indices]
    # Robust endpoints: decorative terminal hooks must not define the glyph.
    trim = max(1, count//10)
    core, directions = points[trim:-trim], tangents[indices][trim:-trim]
    center = np.median(core, axis=0)
    extent = np.ptp(core, axis=0)
    travel = np.abs(deltas).sum(axis=0)
    travel /= max(float(travel.sum()), 1e-10)
    return core, directions, center, extent, travel


def _ordered_cost(a, b, direction_weight):
    """Forward-only path alignment, with signed (not absolute) directions."""
    pa, da, ca, _, _ = a
    pb, db, cb, _, _ = b
    geometry = np.linalg.norm((pa-ca)[:, None, :] - (pb-cb)[None, :, :], axis=2)
    direction = (1-np.clip(da @ db.T, -1, 1))/2
    local = geometry + direction_weight*direction
    previous = [(math.inf, 0)]*(len(pb)+1)
    previous[0] = (0., 0)
    for row in local:
        current = [(math.inf, 0)]
        for j, cost in enumerate(row, 1):
            value, steps = min(previous[j-1], previous[j], current[j-1], key=lambda p: p[0])
            current.append((value+float(cost), steps+1))
        previous = current
    value, steps = previous[-1]
    return value/steps


def _bundle_cost(source, target, config):
    _, _, a, ea, ta = source
    _, _, b, eb, tb = target
    return (.40*float(np.linalg.norm(a-b)) + .20*float(np.linalg.norm(ea-eb))
            + .25*abs(float(ta[0]-tb[0]))
            + .15*_ordered_cost(source, target, config.direction_weight)
            + config.direction_weight*float((1-np.clip(source[1].mean(0) @ target[1].mean(0), -1, 1))/2))


def _assign(source, target, config):
    atoms = _atoms(source, config)
    bundles = _bundles(atoms, source, config)
    repeated_pairs = {tuple(span[0] for span in b.spans) for b in bundles if b.repeated and len(b.spans)==2}
    bundle_sources = [set(span[0] for span in b.spans) for b in bundles]
    # A repeated trace must not be assigned to contradictory reference shapes.
    # Close, distinct reference strokes remain eligible for separate matches.
    reference_paths = [np.array(resample(s,32)) for s in target]
    interchangeable = set()
    for i,a in enumerate(reference_paths):
        for j,b in enumerate(reference_paths[:i]):
            distances = min((np.linalg.norm(a-p,axis=1) for p in (b,b[::-1])),key=lambda d:float(d.mean()))
            if distances.mean()<=config.duplicate_mean_distance+1e-9 and distances.max()<=config.duplicate_max_distance+1e-9:
                interchangeable.add((j,i))
    descriptors = []
    for bundle in bundles:
        pieces = [source[i][a:b+1] for i,a,b in bundle.spans]
        # Compare each repeated trace with the reference, then average its fit.
        # Concatenating repeated traces would introduce an artificial restart.
        descriptors.append([_describe([piece],config.points_per_curve) for piece in pieces]
                           if bundle.repeated else [_describe(pieces,config.points_per_curve)])
    total_ink = sum(a.length for a in atoms)
    candidates = []
    for stroke in target:
        descriptor = _describe([stroke], config.points_per_curve)
        ranked = []
        for index, (bundle, description) in enumerate(zip(bundles, descriptors)):
            fit = sum(_bundle_cost(d, descriptor, config) for d in description)/len(description)
            cost = (fit + config.split_penalty*bundle.cuts
                    + (0. if bundle.repeated else config.merge_penalty*(len(bundle.spans)-1))
                    - config.unused_ink_penalty*bundle.length/total_ink)
            ranked.append((cost, index, fit))
        ranked.sort()
        candidates.append(ranked[:config.candidates_per_target] + [(config.missing_target_cost, None, None)])
    # (objective, used source atoms, selected bundle IDs, pair-fit costs).
    # All reference strokes are solved jointly; no early target owns ink yet.
    beam = [(config.unused_ink_penalty, 0, (), ())]
    for target_index, choices in enumerate(candidates):
        expanded = []
        for cost, used, selected, fits in beam:
            for extra, index, fit in choices:
                mask = bundles[index].mask if index is not None else 0
                if mask & used:
                    continue
                if index is not None and repeated_pairs:
                    current_sources = bundle_sources[index]
                    conflict = False
                    for previous_target, previous in enumerate(selected):
                        if previous is None or (previous_target,target_index) in interchangeable:
                            continue
                        previous_sources = bundle_sources[previous]
                        if any((a in current_sources and b in previous_sources) or
                               (b in current_sources and a in previous_sources) for a,b in repeated_pairs):
                            conflict = True
                            break
                    if conflict:
                        continue
                inversions = sum(bundles[previous].atoms[0] > bundles[index].atoms[0]
                                 for previous in selected if previous is not None) if index is not None else 0
                expanded.append((cost+extra+config.order_penalty*inversions,
                                 used | mask, selected+(index,), fits+(fit,)))
        expanded.sort(key=lambda state: state[0])
        beam = expanded[:config.beam_width]
    best = beam[0]
    return dict(cost=best[0], selected=best[2], fits=best[3], atoms=atoms, bundles=bundles,
                runner_up_cost=beam[1][0] if len(beam)>1 else None,
                candidate_counts=[len(c)-1 for c in candidates])


def _coverage(pieces, target, max_distance):
    """Geometric audit only; union of real projected ink, never a decision gate."""
    starts = np.array(target[:-1]); delta = np.diff(np.array(target), axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    valid = lengths > 1e-10
    starts, delta, lengths = starts[valid], delta[valid], lengths[valid]
    offsets = np.r_[0., np.cumsum(lengths)[:-1]]
    total = float(lengths.sum()); intervals = []
    for piece in pieces:
        for a, b in zip(piece, piece[1:]):
            ink = np.array(b)-a; length = float(np.linalg.norm(ink))
            middle = (np.array(a)+b)/2
            t = np.clip(((middle-starts)*delta).sum(1)/(lengths**2), 0, 1)
            distances = np.linalg.norm(middle-(starts+t[:, None]*delta), axis=1)
            j = int(np.argmin(distances))
            if distances[j] > max_distance:
                continue
            cosine = max(0., float(ink @ delta[j])/max(length*lengths[j], 1e-10))
            center = (offsets[j]+t[j]*lengths[j])/total
            half = length*cosine/(2*total)
            intervals.append((max(0., center-half), min(1., center+half)))
    return _union_length(intervals)


def _assemble(raw_source, source, target, alignment, solution, config):
    scores = _target_scores(solution['fits'], config)
    labels = [[None]*(len(s)-1) for s in source]
    coverage, groups = [], []
    for number, (index, fit) in enumerate(zip(solution['selected'], solution['fits']), 1):
        spans = solution['bundles'][index].spans if index is not None else ()
        for i, a, b in spans:
            labels[i][a:b] = [number]*(b-a)
        pieces = [source[i][a:b+1] for i, a, b in spans]
        coverage.append(_coverage(pieces, target[number-1], config.coverage_distance))
        groups.append(dict(svg_stroke_number=number, fit_cost=fit,
                           assigned=scores['target_assignment'][number-1],
                           similarity_score=scores['target_similarity'][number-1],
                           source_intervals=[dict(source_stroke_number=i+1,
                                                  source_fraction_start=a/(len(source[i])-1),
                                                  source_fraction_end=b/(len(source[i])-1)) for i,a,b in spans]))
    mappings, runs = [], []
    unmapped = total = 0.
    for i, (stroke, assigned) in enumerate(zip(source, labels)):
        for number in sorted({n for n in assigned if n is not None}):
            mappings.append(dict(source_stroke_number=i+1, svg_stroke_number=number,
                                 similarity_score=scores['target_similarity'][number-1]))
        start = 0
        for end in range(1, len(assigned)+1):
            if end < len(assigned) and assigned[end] == assigned[start]:
                continue
            number = assigned[start]
            a, b = start/(len(stroke)-1), end/(len(stroke)-1)
            length = path_length(stroke[start:end+1]); total += length
            if number is None:
                unmapped += length
            connector = number is None and start>0 and end<len(assigned) and assigned[start-1] != assigned[end]
            runs.append(dict(source_stroke_number=i+1, svg_stroke_number=number,
                             sample_start=start, sample_end=end,
                             source_fraction_start=a, source_fraction_end=b,
                             original_start=_source_location(raw_source[i], a),
                             original_end=_source_location(raw_source[i], b),
                             kind='matched' if number else 'connector_candidate' if connector else 'unmapped',
                             reasons=[], reference_intervals=[]))
            start = end
    missing = [g['svg_stroke_number'] for g in groups if not g['source_intervals']]
    return dict(status='mapped', reasons=[], mappings=mappings, proposed_mappings=mappings,
                **scores,
                segment_mappings=runs, target_groups=groups, target_coverage=coverage,
                source_normalized=source, target_normalized=target, alignment=alignment,
                unmatched_fraction=unmapped/max(total, 1e-10), best_cost=solution['cost'],
                runner_up_cost=solution['runner_up_cost'], candidate_counts=solution['candidate_counts'],
                missing_target_strokes=missing, training_usage='review_required',
                quality_flags=['missing_target_strokes'] if missing else [],
                correspondence_basis='target_bundle_joint_exclusive_assignment')


def match_targets(character, raw_source, raw_target, config=TargetMappingConfig()):
    """Return the best complete-glyph hypothesis, even when it may be wrong."""
    if not raw_target or len(raw_target)>33:
        raise ValueError('Expected 1 to 33 ordered SVG strokes')
    target = normalize_strokes(raw_target, 96)
    try:
        source = normalize_strokes(raw_source, 192)
    except ValueError as error:
        return dict(status='invalid', reasons=[str(error)], mappings=[], proposed_mappings=[],
                    **_target_scores([None]*len(target), config),
                    segment_mappings=[], source_normalized=[], target_normalized=target,
                    target_coverage=[0.]*len(target), unmatched_fraction=1.,
                    training_usage='review_required', quality_flags=['invalid_source'])
    source, target, transform = _align(source, target)
    aligned = [('center', source, dict(transform, mode='center'))]
    if config.use_anchor_alignment:
        shift, _ = _anchor_shift(source, target, config)
        if shift is not None:
            shifted = [[tuple(np.array(p)+shift) for p in s] for s in source]
            aligned.append(('first_stroke', shifted, dict(transform, mode='first_stroke', shift_xy=shift.tolist())))
    solutions = [(name, points, alignment, _assign(points, target, config))
                 for name, points, alignment in aligned]
    _, points, alignment, selected = min(solutions, key=lambda entry: entry[3]['cost'])
    result = _assemble(raw_source, points, target, alignment, selected, config)
    result['alignment_hypotheses'] = [dict(mode=name, score=solution['cost'])
                                    for name, _, _, solution in solutions]
    return result
