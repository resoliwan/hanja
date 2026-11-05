"""SCUT-onHCCTestDB TradChar subset 1, first 100 records in original order."""
from paths import DATASETS
import hashlib,json,struct,zipfile
from pathlib import Path
archive=(DATASETS / 'onHCCTestDB-TradChar.zip')
with zipfile.ZipFile(archive) as z:
    base='onHCCTestDB-TradChar/'
    pot=z.read(base+'pot/onHCCTestDB-TradChar 1.pot')
    labels=z.read(base+'label/onHCCTestDB-TradChar 1.txt').decode('utf-8-sig').splitlines()
    samples=[]; offset=0
    for i,label in enumerate(labels[:100]):
        size,code,nstrokes=struct.unpack_from('<HIH',pot,offset)
        decoded=code.to_bytes(4,'little')[:2][::-1].decode('gbk')
        assert decoded==label,(i,decoded,label)
        cursor=offset+8; strokes=[]; stroke=[]
        while True:
            x,y=struct.unpack_from('<hh',pot,cursor);cursor+=4
            if (x,y)==(-1,-1):break
            if (x,y)==(-1,0):
                assert stroke;strokes.append(stroke);stroke=[]
            else:
                assert x>=0 and y>=0;stroke.append([x,y])
        assert cursor==offset+size and len(strokes)==nstrokes and not stroke
        samples.append(dict(id=f'trad1-{i+1:04d}',label=label,strokes=strokes,record=i+1,byte_offset=offset,source='SCUT-onHCCTestDB/TradChar/subset1'))
        offset+=size
(DATASETS / 'tradchar-first100.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2))
(DATASETS / 'sample-manifest.json').write_text(json.dumps({'dataset':'SCUT-onHCCTestDB TradChar (NOT SCUT-COUCH2009 Big5)','source':'https://github.com/HCIILAB/IME_Test','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'selection':'first 100 records, original order, subset 1; no filtering','n':len(samples),'unique_labels':len(set(s['label'] for s in samples)),'pot_label_agreement':100,'labels':''.join(s['label'] for s in samples)},ensure_ascii=False,indent=2))
print((DATASETS / 'sample-manifest.json').read_text())
