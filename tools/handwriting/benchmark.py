"""Run four pretrained engines on labeled ordered stroke JSON; no character conversion."""
from paths import MODELS, RESULTS, ROOT, TOOLS
import argparse, ctypes as C, json, sys, subprocess, time, statistics, csv
from pathlib import Path
from PIL import Image, ImageDraw

sys.path[:0]=[str(ROOT/'vendor/cnn_chinese_hw_legacy'),str(ROOT/'vendor/GlyphWeave')]

def normalize(strokes, size=255, pad=10):
    pts=[p for s in strokes for p in s]
    x0,y0=min(p[0] for p in pts),min(p[1] for p in pts)
    w=max(p[0] for p in pts)-x0; h=max(p[1] for p in pts)-y0
    scale=(size-2*pad)/max(w,h,1)
    return [[[round((x-x0)*scale+(size-w*scale)/2),round((y-y0)*scale+(size-h*scale)/2)] for x,y in s] for s in strokes]

def render(strokes):
    im=Image.new('L',(512,512),255); d=ImageDraw.Draw(im)
    for s in normalize(strokes,511,36):
        points=[tuple(p) for p in s]
        if len(points)>1:d.line(points,fill=0,width=12,joint='curve')
        for x,y in (points[0],points[-1]): d.ellipse((x-6,y-6,x+6,y+6),fill=0)
    return im.resize((128,128),Image.Resampling.LANCZOS)

class Zinnia:
    def __init__(self):
        self.lib=l=C.CDLL(str(MODELS / 'libzinnia.dylib'))
        signatures={
         'recognizer_new':(C.c_void_p,[]),'character_new':(C.c_void_p,[]),
         'recognizer_open':(C.c_int,[C.c_void_p,C.c_char_p]),
         'recognizer_strerror':(C.c_char_p,[C.c_void_p]),
         'character_set_width':(None,[C.c_void_p,C.c_size_t]),'character_set_height':(None,[C.c_void_p,C.c_size_t]),
         'character_add':(C.c_int,[C.c_void_p,C.c_size_t,C.c_int,C.c_int]),
         'recognizer_classify':(C.c_void_p,[C.c_void_p,C.c_void_p,C.c_size_t]),
         'result_size':(C.c_size_t,[C.c_void_p]),'result_value':(C.c_char_p,[C.c_void_p,C.c_size_t]),
         'result_destroy':(None,[C.c_void_p]),'character_destroy':(None,[C.c_void_p]),
         'recognizer_size':(C.c_size_t,[C.c_void_p]),'recognizer_value':(C.c_char_p,[C.c_void_p,C.c_size_t])}
        for name,(ret,args) in signatures.items():
            f=getattr(l,'zinnia_'+name);f.restype=ret;f.argtypes=args
        self.rec=l.zinnia_recognizer_new()
        if not l.zinnia_recognizer_open(self.rec,str(MODELS / 'tegaki-zinnia-traditional-chinese-0.3/handwriting-zh_TW.model').encode()):raise RuntimeError(l.zinnia_recognizer_strerror(self.rec))
        self.classes={l.zinnia_recognizer_value(self.rec,i).decode() for i in range(l.zinnia_recognizer_size(self.rec))}
    def predict(self,strokes):
        l=self.lib;c=l.zinnia_character_new()
        l.zinnia_character_set_width(c,256);l.zinnia_character_set_height(c,256)
        for i,s in enumerate(normalize(strokes)):
            for x,y in s:l.zinnia_character_add(c,i,x,y)
        r=l.zinnia_recognizer_classify(self.rec,c,5)
        if not r:raise RuntimeError('Zinnia classification failed')
        out=[l.zinnia_result_value(r,i).decode() for i in range(l.zinnia_result_size(r))]
        l.zinnia_result_destroy(r);l.zinnia_character_destroy(c)
        return out

def main():
    p=argparse.ArgumentParser();p.add_argument('samples');p.add_argument('--out',default=str(RESULTS / 'benchmark'));a=p.parse_args()
    samples=json.loads(Path(a.samples).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    norm=[dict(s,strokes=normalize(s['strokes'])) for s in samples]
    (out/'normalized.json').write_text(json.dumps(norm,ensure_ascii=False))
    h=json.loads(subprocess.check_output(['node',str(TOOLS / 'hanzi_lookup.cjs'),str(out/'normalized.json')]))
    rows=[];summary={}
    for s,r in zip(samples,h): rows.append(dict(model='Hanzi Lookup',id=s['id'],label=s['label'],**{k:r[k] for k in ['candidates','ms','error']}))
    from cnn_chinese_hw.recognizer.TFLiteRecognizer import TFLiteRecognizer
    from handwriting.inference import Recognizer
    import torch
    torch.set_num_threads(4)
    engines=[('Zinnia',Zinnia()),('cnn_chinese_hw (legacy)',TFLiteRecognizer()),('GlyphWeave',Recognizer(str(MODELS / 'glyphweave-v1.0.0-inference/best.pt'),'cpu'))]
    for name,engine in engines:
        print('Running',name,flush=True)
        for s in samples:
            t=time.perf_counter()
            if name=='Zinnia':c=engine.predict(s['strokes'])
            elif name.startswith('cnn'):c=[chr(int(x[1])) for x in engine.get_candidates_list(s['strokes'],n_cands=5)]
            else:c=[x['character'] for x in engine.predict(render(s['strokes']),topk=5)['candidates']]
            rows.append(dict(model=name,id=s['id'],label=s['label'],candidates=c,ms=(time.perf_counter()-t)*1000,error=None))
        vocab=engine.classes if hasattr(engine,'classes') else [chr(int(x)) for x in engine.classnames]
        summary[name]={'vocabulary_covered':sum(s['label'] in vocab for s in samples)}
    for r in rows:r.update(top1=bool(r['candidates']) and r['label']==r['candidates'][0],top5=r['label'] in r['candidates'])
    for name in dict.fromkeys(r['model'] for r in rows):
        rr=[r for r in rows if r['model']==name]
        summary.setdefault(name,{}).update(n=len(rr),errors=sum(bool(r['error']) for r in rr),top1=sum(r['top1'] for r in rr),top5=sum(r['top5'] for r in rr),median_ms=statistics.median(r['ms'] for r in rr))
    (out/'predictions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (out/'predictions.csv').open('w',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows([dict(r,candidates=' '.join(r['candidates'])) for r in rows])
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    for s in samples:render(s['strokes']).save(out/(s['id'].replace('/','_')+'.png'))
if __name__=='__main__':main()
