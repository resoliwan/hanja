"""Review local correspondence without collapsing an input to one label."""

from html import escape

from .report import _COLORS


def _pieces(row, panel):
    if panel == 'source':
        for run in row['segment_mappings']:
            stroke = row['source_normalized'][run['source_stroke_number']-1]
            yield stroke[run['sample_start']:run['sample_end']+1], run['svg_stroke_number']
    else:
        yield from ((s,i+1) for i,s in enumerate(row['target_normalized']))


def _drawing(row, panel):
    parts = ['<svg viewBox="-.15 -.15 1.3 1.3"><rect x="0" y="0" width="1" height="1" fill="#f8fafc"/>']
    for points, number in _pieces(row,panel):
        color = _COLORS[number-1] if number else '#999'
        path = ' '.join(f'{x:.5f},{y:.5f}' for x,y in points)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width=".012"/>')
    strokes = row['source_normalized'] if panel=='source' else row['target_normalized']
    for i,stroke in enumerate(strokes,1):
        x,y=stroke[0]
        parts.append(f'<circle cx="{x}" cy="{y}" r=".012"/><text x="{x-.025}" y="{y-.025}" font-size=".055">{i}</text>')
    return ''.join(parts)+'</svg>'


def write_segment_report(directory, rows, summary):
    cards=[]
    for row in rows:
        labels = ', '.join(f'{r["source_stroke_number"]}→{r["svg_stroke_number"]}' for r in row['proposed_mappings'])
        covered = ' / '.join(f'{i+1}: {v:.0%}' for i,v in enumerate(row['target_coverage']))
        spans = ''.join(f'<tr><td>{r["source_stroke_number"]}</td><td>{r["source_fraction_start"]:.0%}–{r["source_fraction_end"]:.0%}</td>'
                        f'<td>{r["svg_stroke_number"] or "—"}</td><td>{r["kind"]}</td></tr>' for r in row['segment_mappings'])
        alignment = row.get('alignment',{}).get('mode','center')
        hypotheses = ' / '.join(f'{h["mode"]}: {h["score"]:.3f}' for h in row.get('alignment_hypotheses',[]))
        cards.append(f'''<article data-character="{row['character']}" data-status="{row['status']}">
<h2>{row['character']} {row['sample_id']} · {row['status']}</h2>
<div class="pair">{_drawing(row,'source')}{_drawing(row,'target')}</div>
<p>입력→SVG: {labels or '없음'}</p><p>정답 경로 커버 범위: {covered}</p>
<p>미대응 입력 길이: {row['unmatched_fraction']:.0%} · {escape(', '.join(row['reasons']))}</p>
<p>정렬: {alignment} · {hypotheses}</p>
<details><summary>구간별 대응</summary><table><tr><th>입력</th><th>원본 경로 구간</th><th>SVG</th><th>유형</th></tr>{spans}</table></details></article>''')
    statuses = list(dict.fromkeys(row['status'] for row in rows))
    status_names = {'mapped':'매핑 생성', 'invalid':'입력 오류', 'accepted':'자동 채택', 'pending':'보류'}
    header = '<tr><th>글자</th><th>전체</th>' + ''.join(f'<th>{status_names[s]}</th>' for s in statuses) + '</tr>'
    table=''.join(f'<tr><td>{c}</td><td>{v["total"]}</td>'+''.join(
        f'<td>{sum(r["character"]==c and r["status"]==s for r in rows)}</td>' for s in statuses)+'</tr>'
        for c,v in summary['by_character'].items())
    status_options = ''.join(f'<option value="{s}">{status_names[s]}</option>' for s in statuses)
    options=''.join(f'<option>{c}</option>' for c in summary['by_character'])
    (directory/'review.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>필기와 SVG 획 매핑</title><style>body{font:15px system-ui;max-width:1200px;margin:24px auto;padding:16px;background:#f1f5f9;color:#17233a}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(370px,1fr));gap:16px}article{background:white;padding:16px;border-radius:12px}h2{font-size:18px}.pair{display:flex}.pair svg{width:50%}td,th{padding:6px;border-bottom:1px solid #ddd}select{margin:16px;padding:8px}[hidden]{display:none}</style>
<h1>입력 구간 → 정답 경로 매핑</h1><p>왼쪽은 정렬된 입력, 오른쪽은 SVG입니다. 같은 색이 같은 정답 획을 뜻하며 회색은 미대응/연결 동작 후보입니다.
번호는 각 그림의 획 번호입니다. 입력 하나에 여러 색이 들어갈 수 있습니다.</p>
<p><b>매핑 생성 여부와 커버 범위는 정확도가 아닙니다.</b> 정답 주석과 비교한 평가는 별도 평가 보고서에 기록합니다.
원본 구간 위치는 정규화·재샘플링을 거쳐 근사한 위치입니다. 모든 결과는 학습 전에 검수가 필요합니다. 기존 평가 자료와 모델은 변경하지 않았습니다.</p>
<table>'''+header+table+'''</table>
<select id="character"><option value="">모든 글자</option>'''+options+'''</select><select id="status"><option value="">모든 상태</option>'''+status_options+'''</select>
<div class="grid">'''+''.join(cards)+'''</div><script>for(const id of ['character','status'])document.getElementById(id).onchange=()=>{const c=document.getElementById('character').value,s=document.getElementById('status').value;for(const a of document.querySelectorAll('article'))a.hidden=!!((c&&c!==a.dataset.character)||(s&&s!==a.dataset.status));};</script></html>''',encoding='utf-8')
    _write_sheets(directory,rows)


def _write_sheets(directory, rows):
    from PIL import Image, ImageDraw, ImageFont
    for character in dict.fromkeys(r['character'] for r in rows):
        selected=[r for r in rows if r['character']==character]
        for start in range(0,len(selected),8):
            canvas=Image.new('RGB',(1400,1440),'white'); draw=ImageDraw.Draw(canvas); font=ImageFont.load_default(size=17)
            for k,row in enumerate(selected[start:start+8]):
                x,y=k%2*700,k//2*360
                draw.text((x+10,y+8),f'{row["sample_id"]} {row["status"]} | input / SVG',font=font,fill='black')
                for panel,name in enumerate(('source','target')):
                    for points,number in _pieces(row,name):
                        color=_COLORS[number-1] if number else '#aaa'
                        path=[(x+35+panel*340+250*p[0],y+50+250*p[1]) for p in points]
                        if len(path)>1:draw.line(path,fill=color,width=4)
                    strokes=row['source_normalized'] if panel==0 else row['target_normalized']
                    for i,s in enumerate(strokes,1):
                        sx,sy=x+35+panel*340+250*s[0][0],y+50+250*s[0][1]
                        draw.text((sx,sy-20),str(i),font=font,fill='black')
                text='coverage: '+','.join(f'{v:.0%}' for v in row['target_coverage'])
                draw.text((x+10,y+325),text,font=font,fill='black')
            canvas.save(directory/f'segments-U{ord(character):04X}-{start//8+1}.png')
