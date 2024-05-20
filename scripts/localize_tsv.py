import os
import sys
import os.path as osp
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 1e9

from vlmeval import *

def decode_img(tup):
    im, p = tup
    if osp.exists(p):
        return
    decode_base64_to_image_file(im, p)

def localize_tsv(fname):
    base_name = osp.basename(fname)
    dname = osp.splitext(base_name)[0]
    data = load(fname)
    new_fname = fname.replace('.tsv', '_local.tsv')
    
    indices = list(data['index'])
    images = list(data['image'])
    root = LMUDataRoot()
    root = osp.join(root, 'images', dname)
    os.makedirs(root, exist_ok=True)

    img_paths = [osp.join(root, f'{idx}.jpg') for idx in indices]
    tups = [(im, p) for p, im in zip(img_paths, images)]
    
    pool = mp.Pool(32)
    pool.map(decode_img, tups)
    pool.close()
    data.pop('image')
    data['image_path'] = img_paths
    dump(data, new_fname)

if __name__ == '__main__':
    in_file = sys.argv[1]
    localize_tsv(in_file)