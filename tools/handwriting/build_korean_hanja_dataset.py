"""Build an auditable 3,500-class image dataset; never substitute missing glyphs.

Outputs uncompressed tar shards and a CSV manifest. Fonts and handwriting keep
distinct source types. No training is performed and no validation split is invented.
"""
from paths import DATASETS, RESULTS, ROOT, source_path
import collections
import csv
import hashlib
import io
import json
import struct
import tarfile
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from fontTools.ttLib import TTFont
from benchmark import render


OUT = DATASETS / 'korean-hanja3500-v1'
FIELDS = ['sample_id', 'character', 'class_id', 'codepoint', 'grade', 'source',
          'source_type', 'usage', 'shard', 'member', 'source_reference',
          'group_id', 'pixel_sha256', 'width', 'height', 'variant', 'writer_id']

def digest(data):
    return hashlib.sha256(data).hexdigest()

class PartsReader(io.RawIOBase):
    """Seek over concatenated downloaded zip parts without making a 1 GB copy."""
    def __init__(self, paths):
        self.files = [p.open('rb') for p in paths]
        self.ends = []; total = 0
        for p in paths:
            total += p.stat().st_size; self.ends.append(total)
        self.size = total; self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0: raise ValueError('negative seek')
        return self.pos
    def read(self, size=-1):
        import bisect
        remaining = max(0, self.size-self.pos) if size < 0 else min(size, max(0, self.size-self.pos))
        chunks = []
        while remaining:
            index = bisect.bisect_right(self.ends, self.pos)
            start = self.ends[index-1] if index else 0
            amount = min(remaining, self.ends[index]-self.pos)
            self.files[index].seek(self.pos-start)
            chunk = self.files[index].read(amount)
            if len(chunk) != amount: raise IOError('incomplete archive part')
            chunks.append(chunk); self.pos += amount; remaining -= amount
        return b''.join(chunks)
    def close(self):
        for f in self.files: f.close()
        super().close()

class Writer:
    def __init__(self, source, metadata, counts):
        self.source = source; self.metadata = metadata; self.counts = counts
        self.total = 0; self.tar = None
        self.file = (OUT / 'manifests' / (source+'.csv')).open('w', encoding='utf-8', newline='')
        self.csv = csv.DictWriter(self.file, fieldnames=FIELDS); self.csv.writeheader()
    def add(self, image, character, ref, source_type, usage, group, variant='original'):
        image = image.convert('L')
        assert image.size == (128,128), image.size
        assert image.getextrema()[0] < 220 and image.getextrema()[1] > 230, ref
        if self.total % 10000 == 0:
            if self.tar: self.tar.close()
            self.shard = f'shards/{self.source}-{self.total//10000:04d}.tar'
            self.tar = tarfile.open(OUT / self.shard, 'w')
        key = f'{self.source}-{self.total:07d}'
        member = f'U{ord(character):04X}/{key}.png'
        buf = io.BytesIO(); image.save(buf, format='PNG'); data = buf.getvalue()
        ti = tarfile.TarInfo(member); ti.size = len(data); ti.mtime = 0
        self.tar.addfile(ti, io.BytesIO(data))
        pxhash = digest(image.tobytes()); meta = self.metadata[character]
        self.csv.writerow(dict(sample_id=key, character=character, class_id=meta['class_id'],
            codepoint=f'U+{ord(character):04X}', grade=meta['grade'], source=self.source,
            source_type=source_type, usage=usage, shard=self.shard, member=member,
            source_reference=ref, group_id=group or 'pixels:'+pxhash, pixel_sha256=pxhash,
            width=128, height=128, variant=variant, writer_id='unknown'))
        self.counts[character][self.source] += 1; self.total += 1
    def close(self):
        if self.tar: self.tar.close()
        self.file.close()
        print(self.source, self.total, 'images', flush=True)

def centered(image):
    image = image.convert('L')
    # Only crop white margins; do not change character labels or stroke topology.
    box = image.point(lambda p: 255 if p < 240 else 0).getbbox()
    if box is None: raise ValueError('blank image')
    crop = image.crop(box); scale = 108 / max(crop.size)
    crop = crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.LANCZOS)
    result = Image.new('L',(128,128),255)
    result.paste(crop,((128-crop.width)//2,(128-crop.height)//2))
    return result

def build_fonts(metadata, counts):
    sources = json.loads((DATASETS / 'fonts-korean-hanja/sources.json').read_text())
    coverage = []; writer = Writer('fonts',metadata,counts)
    # Six samples per available font/character; these are NOT six real writers.
    variants = [('base',0,0,0),('rotate_minus3',-3,0,0),('rotate_plus3',3,0,0),
                ('shift_left',0,-3,1),('shift_right',0,3,-1),('smaller',0,0,0)]
    for source in sources:
        path = source_path(source['font']['path']); tt = TTFont(path); cmap = tt.getBestCmap()
        font = ImageFont.truetype(str(path),384)
        missing=[]; available=[]
        for char in metadata:
            glyph = cmap.get(ord(char))
            if glyph is None or tt.getGlyphID(glyph)==0:
                missing.append(char);continue
            canvas=Image.new('L',(768,768),255);d=ImageDraw.Draw(canvas)
            box=d.textbbox((0,0),char,font=font)
            d.text((100-box[0],100-box[1]),char,font=font,fill=0)
            base=centered(canvas);available.append(char)
            for variant,angle,dx,dy in variants:
                im=base.copy()
                if angle: im=im.rotate(angle,Image.Resampling.BICUBIC,fillcolor=255)
                if dx or dy: im=im.transform((128,128),Image.Transform.AFFINE,(1,0,-dx,0,1,-dy),fillcolor=255)
                if variant=='smaller':
                    small=im.resize((116,116),Image.Resampling.LANCZOS)
                    im=Image.new('L',(128,128),255);im.paste(small,(6,6))
                usage='evaluation_only' if source['id']=='noto-sans-kr' else 'train_candidate'
                writer.add(im,char,str(path.relative_to(ROOT)),'synthetic_font',usage,
                    f'font:{source["id"]}:U{ord(char):04X}',variant=source['id']+':'+variant)
        tt.close()
        coverage.append(dict(font=source['id'],covered=len(available),missing=len(missing),
                             missing_characters=missing,covered_characters=available))
        print('Font coverage',source['id'],len(available),'/',len(metadata),flush=True)
    writer.close()
    (OUT/'font-coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2))

def build_aifree(metadata,counts,normalize_label=lambda c: c):
    writer=Writer('ai-free',metadata,counts); invalid=[]
    paths=sorted((DATASETS / 'ai-free').glob('all_data.zip.[0-9][0-9][0-9]'))
    assert len(paths)==12
    with PartsReader(paths) as reader, zipfile.ZipFile(reader,metadata_encoding='cp950') as archive:
        for info in archive.infolist():
            if not info.filename.endswith('.png'):continue
            char=Path(info.filename).name.split('_')[0]
            if len(char)!=1:
                invalid.append(dict(path=info.filename,reason='malformed_label'));continue
            char=normalize_label(char)
            if char not in metadata:continue
            with Image.open(io.BytesIO(archive.read(info))) as image:
                try:
                    normalized=centered(image)
                except ValueError as error:
                    invalid.append(dict(path=info.filename,character=char,reason=str(error)));continue
            writer.add(normalized,char,info.filename,'handwriting_source_unverified','train_candidate',None)
            if writer.total%25000==0:print('AI-FREE processed',writer.total,flush=True)
    writer.close()
    (OUT/'excluded-aifree-filenames.json').write_text(json.dumps(invalid,ensure_ascii=False,indent=2))

def build_scut(metadata,counts,normalize_label=lambda c: c):
    writer=Writer('scut',metadata,counts)
    with zipfile.ZipFile(DATASETS / 'onHCCTestDB-TradChar.zip') as archive:
        for number in range(1,6):
            stem=f'onHCCTestDB-TradChar {number}'
            data=archive.read(f'onHCCTestDB-TradChar/pot/{stem}.pot')
            labels=archive.read(f'onHCCTestDB-TradChar/label/{stem}.txt').decode('utf-8-sig').splitlines()
            offset=0
            for index,label in enumerate(labels):
                size,code,nstrokes=struct.unpack_from('<HIH',data,offset)
                assert code.to_bytes(4,'little')[:2][::-1].decode('gbk')==label
                pos=offset+8;strokes=[];stroke=[]
                while True:
                    x,y=struct.unpack_from('<hh',data,pos);pos+=4
                    if (x,y)==(-1,-1):break
                    if (x,y)==(-1,0):strokes.append(stroke);stroke=[]
                    else:stroke.append([x,y])
                assert pos==offset+size and len(strokes)==nstrokes and not stroke
                offset+=size
                label=normalize_label(label)
                if label in metadata:
                    writer.add(render(strokes),label,f'{stem}.pot:record={index+1}',
                        'recorded_pen_strokes','evaluation_only',f'scut:set{number}:record{index+1}')
            assert offset==len(data)
            print('SCUT set',number,'parsed',len(labels),flush=True)
    writer.close()

def main():
    # Existing output is left untouched: a failed run can be inspected before a new build.
    OUT.mkdir(exist_ok=False)
    (OUT/'shards').mkdir();(OUT/'manifests').mkdir()
    rows=list(csv.DictReader((RESULTS / 'ai-free-3500-coverage/comparison.csv').open(encoding='utf-8-sig')))
    assert len(rows)==3500 and len({r['character'] for r in rows})==3500
    metadata={r['character']:dict(class_id=i,grade=r['introduced_grade']) for i,r in enumerate(sorted(rows,key=lambda r:ord(r['character'])))}
    (OUT/'classes.txt').write_text('\n'.join(metadata)+'\n')
    (OUT/'class-map.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    counts=collections.defaultdict(collections.Counter)
    build_fonts(metadata,counts)
    build_aifree(metadata,counts)
    build_scut(metadata,counts)
    fields=['character','codepoint','class_id','grade','font_count','font_images','ai_free_images','scut_evaluation_images','total_images','has_real_recorded_strokes','has_handwriting_source','font_only','no_images']
    font_cov=json.loads((OUT/'font-coverage.json').read_text())
    font_counts=collections.Counter(c for entry in font_cov for c in entry['covered_characters'])
    coverage=[]
    for char,meta in metadata.items():
        c=counts[char]
        coverage.append(dict(character=char,codepoint=f'U+{ord(char):04X}',**meta,
            font_count=font_counts[char],font_images=c['fonts'],ai_free_images=c['ai-free'],
            scut_evaluation_images=c['scut'],total_images=sum(c.values()),
            has_real_recorded_strokes=bool(c['scut']),has_handwriting_source=bool(c['scut']+c['ai-free']),
            font_only=bool(c['fonts'] and not(c['ai-free']+c['scut'])),no_images=not sum(c.values())))
    with (OUT/'character-coverage.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(coverage)
    summary=dict(target_characters=len(metadata),image_size=[128,128],font_families=5,
        counts_by_source={s:sum(c[s] for c in counts.values()) for s in ['fonts','ai-free','scut']},
        characters_by_source={s:sum(bool(c[s]) for c in counts.values()) for s in ['fonts','ai-free','scut']},
        handwriting_source_characters=sum(r['has_handwriting_source'] for r in coverage),
        font_only_characters=sum(r['font_only'] for r in coverage),no_image_characters=sum(r['no_images'] for r in coverage),
        training_performed=False,scut_usage='evaluation_only',ai_free_provenance='handwriting-derived; original writer identities and synthetic content not independently verified')
    for filename,selected in [('font-only-characters.txt',[r['character'] for r in coverage if r['font_only']]),('no-images-characters.txt',[r['character'] for r in coverage if r['no_images']])]:
        (OUT/filename).write_text('\n'.join(selected)+('\n' if selected else ''))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
