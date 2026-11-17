"""Ordered path geometry shared by correspondence and audit rendering."""

import math


def path_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def resample(points, count):
    if count < 2 or len(points) < 2:
        raise ValueError('invalid_sampling_input')
    cumulative = [0.0]
    for a, b in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(a, b))
    if cumulative[-1] <= 1e-9:
        raise ValueError('zero_length_stroke')
    result, segment = [], 1
    for index in range(count):
        distance = cumulative[-1] * index / (count - 1)
        while segment < len(points) - 1 and cumulative[segment] < distance:
            segment += 1
        span = cumulative[segment] - cumulative[segment - 1]
        t = (distance - cumulative[segment - 1]) / span if span else 0
        result.append(tuple(points[segment - 1][k] * (1-t) + points[segment][k] * t for k in (0, 1)))
    return result


def normalize_strokes(strokes, count=32):
    """Uniformly fit the WHOLE glyph to a unit square, retaining aspect ratio."""
    if count < 2 or not strokes:
        raise ValueError('invalid_sampling_input')
    for stroke in strokes:
        if len(stroke) < 2 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in stroke):
            raise ValueError('invalid_stroke_coordinates')
    points = [p for stroke in strokes for p in stroke]
    low = [min(p[k] for p in points) for k in (0, 1)]
    high = [max(p[k] for p in points) for k in (0, 1)]
    scale = max(high[k] - low[k] for k in (0, 1))
    if scale <= 1e-9:
        raise ValueError('zero_length_stroke')
    center = [(a+b)/2 for a, b in zip(low, high)]
    return [resample([tuple((p[k]-center[k])/scale + 0.5 for k in (0, 1)) for p in stroke], count)
            for stroke in strokes]


def dtw_distance(a, b):
    """Minimum cumulative Euclidean cost, divided by its alignment length."""
    previous = [(math.inf, 0)] * (len(b) + 1)
    previous[0] = (0.0, 0)
    for p in a:
        current = [(math.inf, 0)]
        for j, q in enumerate(b, 1):
            cost, length = min((previous[j-1], previous[j], current[j-1]), key=lambda v: v[0])
            current.append((cost + math.dist(p, q), length + 1))
        previous = current
    cost, length = previous[-1]
    return cost/length


