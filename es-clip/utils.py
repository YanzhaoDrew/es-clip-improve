#!/usr/bin/env python3

import os

# import numpy as np
import torch
import torch.utils
import torchvision

from torchvision.utils import Image

def load_target(fn, resize):
    """
    Load and resize the target image as a NumPy array.

    Args:
        fn (str): The file path of the target image.
        resize (tuple): The desired size of the target image in the format (height, width).

    Returns:
        numpy.ndarray: The preprocessed target image as a NumPy array.
    """
    img = Image.open(fn)
    img = rgba2rgb(img)
    h, w = resize
    img = img.resize((w, h), Image.LANCZOS)
    img_arr = img2tensor(img)
    return img_arr

def img2tensor(img):
    return torchvision.transforms.PILToTensor().__call__(img)

def tensor2img(tensor):
    return torchvision.transforms.ToPILImage().__call__(tensor)

def rgba2rgb(rgba_img):
    h, w = rgba_img.size
    rgb_img = Image.new('RGB', (h, w))
    rgb_img.paste(rgba_img)
    return rgb_img

def infer_height_and_width(hint_height, hint_width, filepath):
    """
    Infers the height and width of an image proportionally, in order to perform scaling without specifying the height and width.

    Parameters:
        hint_height (int): The suggested height value.
        hint_width (int): The suggested width value.
        filepath (str): The path to the image file.

    Returns:
        inferred_height (int): The inferred height value.
        inferred_width (int): The inferred width value.
    """
    fn_width, fn_height = Image.open(filepath).size
    if hint_height <= 0:    # hint_height is invalid
        if hint_width <= 0:
            inferred_height, inferred_width = fn_height, fn_width  # use target image's size
        else:  # hint_width is valid
            inferred_width = hint_width
            inferred_height = hint_width * fn_height // fn_width
    else:  # hint_height is valid
        if hint_width <= 0:
            inferred_height = hint_height
            inferred_width = hint_height * fn_width // fn_height
        else:  # hint_width is valid
            inferred_height, inferred_width = hint_height, hint_width  # use hint size

    print(f'Inferring height and width. '
          f'Hint: {hint_height, hint_width}, File: {fn_width, fn_height}, Inferred: {inferred_height, inferred_width}')

    return inferred_height, inferred_width


def save_as_gif(fn, imgs, fps=24):
    img, *imgs = imgs
    with open(fn, 'wb') as fp_out:
        img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=int(1000./fps), loop=0)

def save_as_frames(fn, imgs, overwrite=True):
    # save to folder `fn` with sequenced filenames
    os.makedirs(fn, exist_ok=True)
    for i, img in enumerate(imgs):
        this_fn = os.path.join(fn, f'{i:08}.png')
        if overwrite or not os.path.exists(this_fn):
            save_as_png(this_fn, img)

def save_as_png(fn, img):
    if not fn.endswith('.png'):
        fn = f'{fn}.png'
    img.save(fn)

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__  # type: ignore 
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# Copied from https://github.com/makinacorpus/easydict/blob/master/easydict/__init__.py
class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)
