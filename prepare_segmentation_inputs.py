"""Prepare a new input directory using de-identified release keys only."""
import argparse, csv, io
from pathlib import Path
from zipfile import ZipFile

def prepare(analysis, originals, target):
    target.mkdir(parents=True,exist_ok=False)
    with ZipFile(analysis) as a, ZipFile(originals) as o:
        rows=list(csv.DictReader(io.TextIOWrapper(a.open('features.csv'),encoding='utf8')))
        out=[]
        for r in rows:
            dog,side=r['ID'],r['R/L']; key=r['eye_key']; annotated=int(r['label'])
            if len(dog)!=4 or dog[0]!='D' or not dog[1:].isascii() or not dog[1:].isdigit() or side not in ['L','R'] or key!=f'{dog}_{side}':
                raise ValueError('Invalid release key')
            number=int(dog[1:]); filename=f'{number:04d}_{side}_01.png'
            source=o if annotated else a
            name=f'original_resolution_images/{key}.png' if annotated else f'images/{key}.png'
            folder='labeled_image/image' if annotated else 'all_img'
            p=target/folder/filename; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(source.read(name))
            if annotated:
                p=target/'labeled_image/label'/filename; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(a.read(f'manual_masks/{key}.png'))
            out.append([number,side,r['fold'],annotated])
        with (target/'dataset.csv').open('w',encoding='utf8',newline='') as f:
            w=csv.writer(f); w.writerow(['test_ID','R/L','fold','label']);w.writerows(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis_archive',type=Path);p.add_argument('originals_archive',type=Path);p.add_argument('new_inputs',type=Path)
    a=p.parse_args();prepare(a.analysis_archive,a.originals_archive,a.new_inputs)
