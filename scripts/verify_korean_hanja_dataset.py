"""Decode every sample, reconcile manifests, audit duplicates and make a preview."""
import collections,csv,hashlib,io,json,shutil,tarfile
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/korean-hanja3500-v1'

def main():
    classes=(OUT/'classes.txt').read_text().splitlines()
    assert len(classes)==len(set(classes))==3500
    target=set(classes); counts=collections.defaultdict(collections.Counter)
    hash_labels=collections.defaultdict(set); hash_sources=collections.defaultdict(set)
    hashes=collections.Counter(); samples=collections.Counter(); total=0
    hash_usages=collections.defaultdict(set)
    # Show difficult/new characters alongside ordinary ones for each source.
    preview_chars=list('學東門靑馬魚㥠䄷乭畓')
    fonts=json.loads((ROOT/'data/fonts-korean-hanja/sources.json').read_text())
    sources=[s['id'] for s in fonts]+['ai-free','scut']; selected={}
    for manifest in sorted((OUT/'manifests').glob('*.csv')):
        with manifest.open() as f:
            groups=collections.defaultdict(list)
            for row in csv.DictReader(f):groups[row['shard']].append(row)
        for shard,rows in groups.items():
            with tarfile.open(OUT/shard) as tar:
                members=tar.getmembers();assert len(members)==len(rows)
                for member,row in zip(members,rows):
                    assert member.name==row['member']
                    with Image.open(io.BytesIO(tar.extractfile(member).read())) as im:
                        im.load();assert im.mode=='L' and im.size==(128,128)
                        assert hashlib.sha256(im.tobytes()).hexdigest()==row['pixel_sha256']
                        assert im.getextrema()[0]<220 and im.getextrema()[1]>230
                        # A white frame is needed to avoid clipped glyphs.
                        edges=[im.crop(box).getextrema()[0] for box in [(0,0,128,1),(0,127,128,128),(0,0,1,128),(127,0,128,128)]]
                        assert min(edges)>=230, (row['sample_id'],'clipped frame',edges)
                        char=row['character'];assert char in target and classes[int(row['class_id'])]==char
                        source=row['source'];counts[char][source]+=1;samples[source]+=1;total+=1
                        h=row['pixel_sha256'];hash_labels[h].add(char);hash_sources[h].add(source);hashes[h]+=1
                        hash_usages[h].add(row['usage'])
                        visual_source=row['variant'].split(':')[0] if source=='fonts' else source
                        key=(visual_source,char)
                        if char in preview_chars and key not in selected and (source!='fonts' or row['variant'].endswith(':base')):
                            selected[key]=im.copy()
            print('Verified',shard,len(rows),flush=True)
    with (OUT/'character-coverage.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            c=counts[row['character']]
            assert c['fonts']==int(row['font_images']) and c['ai-free']==int(row['ai_free_images']) and c['scut']==int(row['scut_evaluation_images'])
    conflicts=[dict(pixel_sha256=h,characters=sorted(labels),sources=sorted(hash_sources[h])) for h,labels in hash_labels.items() if len(labels)>1]
    cross_eval=[h for h,s in hash_usages.items() if 'evaluation_only' in s and 'train_candidate' in s]
    (OUT/'conflicting-label-pixels.json').write_text(json.dumps(conflicts,ensure_ascii=False,indent=2))
    (OUT/'evaluation-overlap-hashes.txt').write_text('\n'.join(cross_eval)+('\n' if cross_eval else ''))
    # A safe training loader can exclude conflicting labels and exact evaluation overlaps.
    excluded=set(cross_eval)|{x['pixel_sha256'] for x in conflicts}
    (OUT/'training-excluded-pixel-hashes.txt').write_text('\n'.join(sorted(excluded))+('\n' if excluded else ''))
    dup_by_source={}
    for source in samples:
        unique=sum(source in ss for ss in hash_sources.values())
        dup_by_source[source]=dict(samples=samples[source],unique_pixel_images=unique,duplicate_samples=samples[source]-unique)
    verification=dict(verified_images=total,all_pngs_decoded=True,all_pixel_hashes_verified=True,
        all_class_ids_verified=True,all_frames_unclipped=True,source_counts=dict(samples),
        exact_duplicates=dup_by_source,conflicting_label_pixel_groups=len(conflicts),
        exact_train_evaluation_overlap_groups=len(cross_eval),excluded_training_pixel_groups=len(excluded))
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    cell=112;left=170;top=85
    sheet=Image.new('RGB',(left+len(preview_chars)*cell,top+len(sources)*cell),'#edf1f5');d=ImageDraw.Draw(sheet)
    title=ImageFont.truetype(str(ROOT/fonts[0]['font']['path']),25)
    label=ImageFont.truetype(str(ROOT/fonts[0]['font']['path']),15)
    d.text((16,12),'Korean Hanja 3500 — fonts / AI-FREE / SCUT',font=title,fill='black')
    for j,c in enumerate(preview_chars):d.text((left+j*cell+38,50),c,font=title,fill='black')
    for i,s in enumerate(sources):
        d.text((10,top+i*cell+40),s,font=label,fill='black')
        for j,c in enumerate(preview_chars):
            x,y=left+j*cell,top+i*cell
            if (s,c) in selected:sheet.paste(selected[s,c].resize((104,104)),(x,y))
            else:d.text((x+12,y+40),'NO DATA',font=label,fill='#a22626')
    sheet.save(OUT/'preview.png')
    license_dir=OUT/'source-documents';license_dir.mkdir(exist_ok=True)
    for font in fonts:
        shutil.copyfile(ROOT/font['license']['path'],license_dir/(font['id']+'-LICENSE.txt'))
    shutil.copyfile(ROOT/'data/fonts-korean-hanja/sources.json',license_dir/'font-sources.json')
    shutil.copyfile(ROOT/'data/ai-free/download-manifest.json',license_dir/'ai-free-download-manifest.json')
    shutil.copyfile(ROOT/'data/ai-free/README-pinned.md',license_dir/'AI-FREE-README.md')
    shutil.copyfile(ROOT/'data/ime-readme.md',license_dir/'SCUT-IME-Test-README.md')
    with (OUT/'SHA256SUMS.txt').open('w') as f:
        for path in sorted((OUT/'shards').glob('*.tar')):
            h=hashlib.sha256()
            with path.open('rb') as source:
                while block:=source.read(1024*1024):h.update(block)
            f.write(h.hexdigest()+'  '+str(path.relative_to(OUT))+'\n')
    print(json.dumps(verification,indent=2),flush=True)

if __name__=='__main__':main()
