from common import functools_ext as ft
from torch.nn import functional as F


def pad(img, fac, mode='replicate'):
    """
    pad img such that height and width are divisible by fac
    """
    _, _, h, w = img.shape
    padH = fac - (h % fac)
    padW = fac - (w % fac)
    if padH == fac and padW == fac:
        return img, ft.identity
    if padH == fac:
        padTop = 0
        padBottom = 0
    else:
        padTop = padH // 2
        padBottom = padH - padTop
    if padW == fac:
        padLeft = 0
        padRight = 0
    else:
        padLeft = padW // 2
        padRight = padW - padLeft
    assert (padTop + padBottom + h) % fac == 0
    assert (padLeft + padRight + w) % fac == 0

    def _undo_pad(img_):
        # the or None makes sure that we don't get 0:0
        img_out = img_[..., padTop:(-padBottom or None), padLeft:(-padRight or None)]
        assert img_out.shape[-2:] == (h, w), (img_out.shape[-2:], (h, w), img_.shape,
                                              (padLeft, padRight, padTop, padBottom))
        return img_out

    return F.pad(img, (padLeft, padRight, padTop, padBottom), mode), _undo_pad
