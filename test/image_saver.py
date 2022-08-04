import glob
import os

import torch
from PIL import Image

from vis.image_summaries import to_image


class ImageSaver(object):
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.saved_fs = []

    def __str__(self):
        return 'ImageSaver({})'.format(self.out_dir)

    def save_img(self, img, filename, convert_to_image=True):
        """
        :param img: image tensor, in {0, ..., 255}
        :param filename: output filename
        :param convert_to_image: if True, call to_image on img, otherwise assume this has already been done.
        :return:
        """
        if convert_to_image:
            img = to_image(img.type(torch.uint8))
        out_p = self.get_save_p(filename)
        Image.fromarray(img).save(out_p)
        return out_p

    def get_save_p(self, file_name):
        out_p = os.path.join(self.out_dir, file_name)
        self.saved_fs.append(file_name)
        return out_p

    def file_starting_with_exists(self, prefix):
        check_p = os.path.join(self.out_dir, prefix) + '*'
        print(check_p, glob.glob(check_p))
        return len(glob.glob(check_p)) > 0
