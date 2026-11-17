"""One SVG -> ten deterministic cases and provenance-based correspondence labels."""

import hashlib
import json
import random
import math
from pathlib import Path

from .geometry import normalize_strokes, path_length
from .sources import read_svg_strokes


def generate_svg_cases(svg_path: Path):
    digest = hashlib.sha256(svg_path.read_bytes()).hexdigest()
    rng = random.Random(int(digest[:16],16))
    raw = read_svg_strokes(svg_path)
    template = [[[(x-.5)*.7+.5,(y-.5)*.7+.5] for x,y in s]
                for s in normalize_strokes(raw,33)]
    character = chr(int(svg_path.stem[1:],16))
    n = len(template)
    baseline = [[i+1]*(len(s)-1) for i,s in enumerate(template)]
    cases = []

    def add(name, strokes, labels, policy='segment_correspondence'):
        assert len(strokes)==len(labels)
        assert all(len(s)-1==len(l) for s,l in zip(strokes,labels))
        pairs = sorted({(i+1,j) for i,ls in enumerate(labels) for j in ls if j})
        cases.append(dict(case_id=f'U{ord(character):04X}-{name}', character=character,
                          variant=name, template=template, strokes=strokes,
                          expected_relations=pairs, expected_edge_labels=labels,
                          evaluation_policy=policy, svg_sha256=digest,
                          generator_version=2))

    add('original',template,baseline)
    add('position_scale',[[[(x-.5)*.85+.55,(y-.5)*.85+.46] for x,y in s] for s in template],baseline)
    add('local_jitter',[[[x+(.025 if i==0 else 0)+rng.uniform(-.006,.006),
                          y+rng.uniform(-.006,.006)] for x,y in s] for i,s in enumerate(template)],baseline)
    add('reverse_direction',[list(reversed(s)) if i%2==0 else s for i,s in enumerate(template)],baseline)
    order=list(reversed(range(n)))
    add('reverse_order',[template[i] for i in order],[baseline[i] for i in order])
    if n>=2:
        add('join',[template[0]+template[1]]+template[2:],
            [baseline[0]+[None]+baseline[1]]+baseline[2:])
    else:
        # A single-stroke glyph has no two distinct reference strokes to join.
        # Split and reconnect its own two halves without inserting new ink.
        middle=len(template[0])//2
        joined=template[0][:middle+1]+template[0][middle+1:]
        add('join',[joined],baseline)
    longest=max(range(n),key=lambda i:path_length(template[i]))
    middle=len(template[longest])//2
    add('split',template[:longest]+[template[longest][:middle+1],template[longest][middle:]]+template[longest+1:],
        baseline[:longest]+[baseline[longest][:middle],baseline[longest][middle:]]+baseline[longest+1:])
    # Fixed negative fixture needs no other SVG or batch-dependent donor pool.
    # This measures whole-character rejection, not shared-stroke recognition.
    foreign = [[[.25,.5],[.75,.5]],[[.5,.2],[.5,.8]]]
    if character=='十':
        foreign=[[[.2,.3],[.8,.3]],[[.2,.7],[.8,.7]]]
    add('different_character',foreign,[[None]*(len(s)-1) for s in foreign],
        policy='whole_character_rejection')
    extra=[[.08,.08],[.12,.10],[.08,.14],[.12,.18]]
    add('extra_ink',template+[extra],baseline+[[None]*(len(extra)-1)])
    phase=rng.uniform(0,2*math.pi)
    duplicate=[[x+.006+.002*math.sin(phase+i*.3),y-.004+.002*math.cos(phase+i*.3)]
               for i,(x,y) in enumerate(template[0])]
    add('duplicate',template+[duplicate],baseline+[baseline[0]])
    assert len(cases)==10
    # Return independent JSON-compatible data, never shared mutable strokes.
    return json.loads(json.dumps(cases))
