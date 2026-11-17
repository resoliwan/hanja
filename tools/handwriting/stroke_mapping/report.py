"""Self-contained review page for generated correspondences (no remote assets)."""

from collections import Counter
from html import escape


_COLORS = ('#2563eb', '#ea580c', '#16945a', '#9333ea', '#c02662')


def _drawing(strokes, numbers):
    elements = ['<svg viewBox="-0.12 -0.12 1.24 1.24" role="img">',
                '<rect x="0" y="0" width="1" height="1" fill="#f8fafc" stroke="#ddd" stroke-width=".005"/>']
    for index, points in enumerate(strokes):
        number = numbers.get(index + 1)
        color = _COLORS[number-1] if number else '#888'
        path = ' '.join(f'{x:.5f},{y:.5f}' for x, y in points)
        elements.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width=".014" stroke-linecap="round" stroke-linejoin="round"/>')
        x, y = points[0]
        elements.append(f'<circle cx="{x}" cy="{y}" r=".018" fill="{color}"/>')
        elements.append(f'<text x="{x-.025}" y="{y-.035}" font-size=".065" fill="{color}">{index+1}</text>')
    elements.append('</svg>')
    return ''.join(elements)


def write_report(path, rows, summary):
    cards = []
    for row in rows:
        proposed = {p['source_stroke_number']: p['svg_stroke_number'] for p in row['proposed_mappings']}
        target_numbers = {i+1: i+1 for i in range(len(row['target_normalized']))}
        source = _drawing(row['source_normalized'], proposed)
        target = _drawing(row['target_normalized'], target_numbers)
        mapping = ', '.join(f'{i}→{j}' for i, j in proposed.items()) or '없음'
        reasons = ', '.join(row['reasons']) or '자동 채택 조건 충족'
        score = f'{row["best_cost"]:.4f}' if row['best_cost'] is not None else '—'
        margin = f'{row["assignment_margin"]:.4f}' if row['assignment_margin'] is not None else '— (구조적 정렬)'
        stability = f'{row["stability"]:.0%}' if row.get('stability') is not None else '—'
        quality = ', '.join(row.get('quality_flags', [])) or '특이사항 없음'
        cards.append(f'''<article class="{row['status']}" data-character="{row['character']}" data-status="{row['status']}">
<h2>{row['character']} · {escape(row['sample_id'])} <small>{row['status']}</small></h2>
<div class="pair"><figure>{source}<figcaption>SCUT · 번호는 실제 작성 순서</figcaption></figure>
<figure>{target}<figcaption>SVG · 번호는 정답 획순</figcaption></figure></div>
<p>입력→SVG: <b>{mapping}</b></p><p>차이 점수 {score} · 차선 배정과의 차이 {margin}</p>
<p>{escape(reasons)}</p><p>변형 검사 일치율 {stability} (정답 확률 아님)</p>
<p>필체 특성: {escape(quality)} · 학습 사용: {escape(row.get('training_usage', 'review_required'))}</p></article>''')
    counts = ''.join(f'<tr><td>{c}</td><td>{v["total"]}</td><td>{v["accepted"]}</td><td>{v["pending"]}</td></tr>'
                     for c, v in summary['by_character'].items())
    path.write_text('''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>一·二·三·用 획 매핑 검토</title>
<style>body{font:15px system-ui;margin:28px auto;padding:0 18px;max-width:1100px;color:#18243a;background:#f3f5f8}
h1{font-size:26px}h2{font-size:18px}small{float:right;color:#475569}p{line-height:1.6}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
article{background:white;padding:16px;border-radius:12px;border-top:4px solid #16945a}
article.pending{border-color:#ea580c}.pair{display:flex}figure{margin:0;width:50%}svg{width:100%}figcaption{font-size:12px;text-align:center}
table{border-collapse:collapse;margin:20px 0}td,th{padding:8px 24px;border-bottom:1px solid #ccc;text-align:left}
select{padding:8px;margin:8px 10px 20px 0}[hidden]{display:none}</style>
<h1>一·二·三·用 — 완성된 필기의 획별 답안지</h1>
<p>같은 색은 대응하는 획입니다. 점은 획의 시작점입니다. 두 그림은 각각 글자 전체를 기준으로 정규화했습니다.
보류 샘플의 색은 최선의 <b>제안</b>이며 학습 정답으로 채택하지 않았습니다.</p>
<p><b>자동 생성한 후보 정답입니다. 채택률은 정확도가 아닙니다.</b> 사람이 확인한 정답과 비교한 정확도는 아직 측정하지 않았습니다.
원래 SCUT 평가용 분할은 변경하지 않았고 모델 학습도 실행하지 않았습니다.
획 대응과 필체 특성을 별도로 기록합니다. 대응이 채택되어도 학습 적합성 검토가 필요할 수 있습니다.
작성 순서는 매핑 점수에 사용하지 않습니다.</p>
<table><thead><tr><th>글자</th><th>전체</th><th>자동 채택</th><th>보류</th></tr></thead><tbody>'''
                    + counts + '''</tbody></table>
<select id="character" aria-label="글자"><option value="">모든 글자</option><option>一</option><option>二</option><option>三</option><option>用</option></select>
<select id="status" aria-label="상태"><option value="">모든 상태</option><option value="accepted">자동 채택</option><option value="pending">보류</option></select>
<div class="grid">''' + '\n'.join(cards) + '''</div>
<script>for(const id of ['character','status']) document.getElementById(id).onchange=()=>{
const c=document.getElementById('character').value,s=document.getElementById('status').value;
for(const card of document.querySelectorAll('article')) card.hidden=!!((c&&card.dataset.character!==c)||(s&&card.dataset.status!==s));
};</script></html>''', encoding='utf-8')


def summarize(rows):
    result = {}
    for character in dict.fromkeys(r['character'] for r in rows):
        selected = [r for r in rows if r['character'] == character]
        counts = Counter(r['status'] for r in selected)
        result[character] = dict(total=len(selected), accepted=counts['accepted'], pending=counts['pending'])
        result[character]['status_counts'] = dict(counts)
        result[character]['unique_trajectories'] = len({r.get('trajectory_sha256', r['sample_id']) for r in selected})
    return dict(
        total=len(rows), by_character=result, status_counts=dict(Counter(r['status'] for r in rows)),
        unique_trajectories=len({r.get('trajectory_sha256', r['sample_id']) for r in rows}),
        pending_reasons=dict(Counter(reason for r in rows for reason in r['reasons'])),
        training_usage=dict(Counter(r.get('training_usage') for r in rows)),
        accuracy=None, accuracy_note='No independently annotated correspondence test set; acceptance is not accuracy.',
        usage='pseudo_label_candidates_only; original SCUT evaluation splits unchanged',
    )


def write_overview(path, rows):
    """Export the same proposed assignments for quick inspection outside HTML."""
    from PIL import Image, ImageDraw, ImageFont

    columns, width, height = 4, 380, 235
    canvas = Image.new('RGB', (columns*width, 60 + ((len(rows)+columns-1)//columns)*height), '#f3f5f8')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    draw.text((16, 12), 'SCUT -> SVG: provisional stroke correspondence (not measured accuracy)', fill='#18243a', font=font)
    draw.text((16, 33), 'Left: handwriting / Right: SVG. Colors = SVG stroke IDs. Dots = starts.', fill='#18243a', font=font)
    for index, row in enumerate(rows):
        x, y = (index % columns)*width, 60 + (index // columns)*height
        draw.rounded_rectangle((x+5, y+5, x+width-5, y+height-5), radius=8, fill='white')
        color = '#16945a' if row['status'] == 'accepted' else '#ea580c'
        draw.text((x+12, y+12), f'{row["sample_id"]}  U+{ord(row["character"]):04X}  {row["status"]}', fill=color, font=font)
        mapping = {m['source_stroke_number']: m['svg_stroke_number'] for m in row['proposed_mappings']}
        for panel, strokes in enumerate((row['source_normalized'], row['target_normalized'])):
            for stroke_index, stroke in enumerate(strokes, 1):
                number = mapping.get(stroke_index) if panel == 0 else stroke_index
                stroke_color = _COLORS[number-1] if number else '#888'
                points = [(x+22+panel*185+140*p[0], y+48+140*p[1]) for p in stroke]
                draw.line(points, fill=stroke_color, width=3)
                sx, sy = points[0]
                draw.ellipse((sx-3, sy-3, sx+3, sy+3), fill=stroke_color)
                draw.text((sx, sy-19), str(stroke_index), fill=stroke_color, font=font)
        label = ', '.join(f'{i}->{j}' for i, j in mapping.items()) or 'No assignment (count/geometry)'
        draw.text((x+12, y+207), label, fill='#18243a', font=font)
    canvas.save(path)


def write_visual_sheets(directory, rows, templates):
    """Raw trajectories without algorithm scores/colors, for visual assessment."""
    from PIL import Image, ImageDraw, ImageFont
    for character in dict.fromkeys(r['character'] for r in rows):
        selected = [r for r in rows if r['character'] == character]
        for start in range(0, len(selected), 8):
            image = Image.new('RGB', (1400, 1440), 'white')
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default(size=18)
            for index, row in enumerate(selected[start:start+8]):
                x, y = index % 2*700, index//2*360
                draw.text((x+12, y+8), row['sample_id']+f' U+{ord(character):04X} | SCUT / SVG', fill='black', font=font)
                for panel, strokes in enumerate((row['strokes'], templates[character])):
                    # Normalize the coordinate frame, but draw ALL raw points.
                    points = [p for s in strokes for p in s]
                    low = [min(p[k] for p in points) for k in (0, 1)]
                    high = [max(p[k] for p in points) for k in (0, 1)]
                    scale = max(high[k]-low[k] for k in (0, 1)) or 1
                    center = [(a+b)/2 for a, b in zip(low, high)]
                    for i, stroke in enumerate(strokes, 1):
                        path = [(x+30+panel*340+260*((p[0]-center[0])/scale+.5),
                                 y+55+260*((p[1]-center[1])/scale+.5)) for p in stroke]
                        if len(path) > 1:
                            draw.line(path, fill='black', width=3)
                        sx, sy = path[0]
                        draw.ellipse((sx-3, sy-3, sx+3, sy+3), fill='black')
                        draw.text((sx, sy-23), str(i), fill='black', font=font)
            image.save(directory / f'visual-U{ord(character):04X}-{start//8+1}.png')
