import json
import pathlib
import zlib
import argparse
import glob
import os

from PIL import Image
import numpy as np
import torch

parser = argparse.ArgumentParser(
    prog='PNG2EMBED',
    description='Convert png files back to their original embeddings')

parser.add_argument('--image', required=False)
parser.add_argument('--folder', required=False)
args = parser.parse_args()


class EmbeddingDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, d):
        if 'TORCHTENSOR' in d:
            return torch.from_numpy(np.array(d['TORCHTENSOR']))
        return d


def crop_black(img, tol=0):
    mask = (img > tol).all(2)
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), mask.shape[1] - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), mask.shape[0] - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def lcg(m=2 ** 32, a=1664525, c=1013904223, seed=0):
    while True:
        seed = (a * seed + c) % m
        yield seed % 255


def xor_block(block):
    g = lcg()
    rand_block = np.array([next(g) for _ in range(np.product(block.shape))]).astype(np.uint8).reshape(block.shape)
    return np.bitwise_xor(block.astype(np.uint8), rand_block & 0x0F)


def extract_image_data_embed(image):
    d = 3
    out_arr = crop_black(
        np.array(image.convert('RGB').getdata()).reshape(image.size[1], image.size[0], d).astype(np.uint8)) & 0x0F
    black_cols = np.where(np.sum(out_arr, axis=(0, 2)) == 0)
    if black_cols[0].shape[0] < 2:
        print('No Image data blocks found.')
        return None

    data_block_lower = out_arr[:, :black_cols[0].min(), :].astype(np.uint8)
    data_block_upper = out_arr[:, black_cols[0].max() + 1:, :].astype(np.uint8)

    data_block_lower = xor_block(data_block_lower)
    data_block_upper = xor_block(data_block_upper)

    data_block = (data_block_upper << 4) | data_block_lower
    data_block = data_block.flatten().tobytes()

    data = zlib.decompress(data_block)
    return json.loads(data, cls=EmbeddingDecoder)


if args.image:
    fp = pathlib.Path(args.image)
    parent = fp.parent
    os.chdir(parent)
    all_data = extract_image_data_embed(Image.open(fp.name))
    trimmed = {k: v for k, v in all_data.items() if k == 'string_to_token' or k == 'string_to_param'}
    torch.save(trimmed, open(f'{fp.name.split(".")[0]}.pt', 'wb'))
elif args.folder:
    to_process = []
    os.chdir(args.folder)
    os.makedirs('converted')
    to_process += glob.glob('*.png')
    for image in to_process:
        all_data = extract_image_data_embed(Image.open(image))
        trimmed = {k: v for k, v in all_data.items() if k == 'string_to_token' or k == 'string_to_param'}
        torch.save(trimmed, open(f'converted/{image.split(".")[0]}.pt', 'wb'))
