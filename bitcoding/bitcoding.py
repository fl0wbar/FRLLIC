import os

import numpy as np
import torch
from common import functools_ext as ft

import bitcoding.coders_helpers
import pytorch_ext as pe
from bitcoding.coders import ArithmeticCoder
from test import cuda_timer


# A random sequence of bytes used to separate the bitstreams of different scales
_MAGIC_VALUE_SEP = b'\x46\xE2\x84\x92'


class Bitcoding(object):
    """
    Class to encode an image to a file and decode it. Saves timings of individual steps to `times`.
    If `compare_with_theory = True`, also compares actual bitstream size to size predicted by cross entropy. Note
    that this is slower because we need to evaluate the loss.
    """
    def __init__(self, blueprint, times: cuda_timer.StackTimeLogger, compare_with_theory=False):
        self.blueprint = blueprint
        self.compare_with_theory = compare_with_theory
        self.times = times

    def encode(self, img, pout):
        """
        Encode image to disk at path `p`.
        :param img: uint8 tensor of shape CHW or 1CHW
        :param pout: path
        """
        assert not os.path.isfile(pout)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)  # 1CHW
        assert len(img.shape) == 4 and img.shape[0] == 1 and img.shape[1] == 3, img.shape
        assert img.dtype == torch.int64, img.dtype

        img = img.float()

        with self.times.run('[-] encode forwardpass'):
            out = self.blueprint.net(img)

        if self.compare_with_theory:
            with self.times.run('[-] get loss'):
                loss_out = self.blueprint.get_loss(out)

        self.blueprint.net.zero_grad()

        entropy_coding_bytes = []  # bytes used by different scales

        with open(pout, 'wb') as fout:
            for scale, dmll, uniform in self.iter_scale_dmll():
                with self.times.prefix_scope(f'[{scale}]'):
                    if uniform:
                        entropy_coding_bytes.append(
                                self.encode_uniform(dmll, out.S[scale], fout))
                    else:
                        entropy_coding_bytes.append(
                                self.encode_scale(scale, dmll, out, img, fout))
                    fout.write(_MAGIC_VALUE_SEP)

        num_subpixels = np.prod(img.shape)
        actual_num_bytes = os.path.getsize(pout)
        actual_bpsp = actual_num_bytes * 8 / num_subpixels

        if self.compare_with_theory:
            assumed_bpsps = [b * 8 / num_subpixels for b in entropy_coding_bytes]
            tostr = lambda l: ' | '.join(map('{:.3f}'.format, l)) + f' => {sum(l):.3f}'
            overhead = (sum(assumed_bpsps) / sum(loss_out.nonrecursive_bpsps) - 1) * 100
            return f'Bitrates:\n' \
                f'theory:  {tostr(loss_out.nonrecursive_bpsps)}\n' \
                f'assumed: {tostr(list(reversed(assumed_bpsps)))} [{overhead:.2f}%]\n' \
                f'actual:                                => {actual_bpsp:.3f} [{actual_num_bytes} bytes]'
        else:
            return actual_bpsp

    def decode(self, pin):
        """
        :param pin:  Path where image is stored
        :return: Decoded image, as 1CHW, long
        """
        with open(pin, 'rb') as fin:
            dec_out_prev = None
            bn_prev = None  # bn of bottleneck - 1

            for scale, dmll, uniform in self.iter_scale_dmll():
                with self.times.prefix_scope(f'[{scale}]'):
                    if uniform:
                        assert bn_prev is None
                        S_prev = self.decode_uniform(dmll, fin)
                        bn_prev = dmll.to_bn(S_prev)
                    else:
                        with self.times.run('get_P'):
                            l, dec_out_prev = self.blueprint.net.get_P(
                                    scale, bn_prev, dec_out_prev)
                        bn_prev = self.decode_scale(dmll, l, fin)
                    assert fin.read(4) == _MAGIC_VALUE_SEP  # assert valid file

            assert bn_prev is not None  # assert decoding worked

        return bn_prev.round().long()

    def iter_scale_dmll(self):
        """ from smallest to largest
         yields (scale, loss_dmm, uniform) """
        for scale in reversed(range(self.blueprint.net.scales + 1)):
            yield (scale,
                   self.blueprint.losses.loss_dmol_rgb if scale == 0 else self.blueprint.losses.loss_dmol_n,
                   scale == self.blueprint.net.scales)

    def encode_uniform(self, dmll, S, fout):
        """ encode coarsest scale, for which we assume a uniform prior. """
        write_shape(S.shape, fout)
        r = ArithmeticCoder(dmll.L)

        entropy_coding_bytes = 0
        with self.times.prefix_scope('uniform encode'):
            c_uniform = self._get_uniform_cdf(S.shape, dmll.L)
            for c in range(S.shape[1]):
                S_c = S[:, c, ...].to(torch.int16)
                encoded = r.range_encode(S_c, c_uniform, self.times)
                write_num_bytes_encoded(len(encoded), fout)
                entropy_coding_bytes += len(encoded)
                fout.write(encoded)

        return entropy_coding_bytes

    def decode_uniform(self, dmll, fin):
        """ decode coarsest scale, for which we assume a uniform prior. """
        C, H, W = read_shapes(fin)
        r = ArithmeticCoder(dmll.L)

        S = []
        with self.times.prefix_scope('uniform decode'):
            c_uniform = self._get_uniform_cdf((1, C, H, W), dmll.L)
            for c in range(C):
                num_bytes = read_num_bytes_encoded(fin)
                encoded = fin.read(num_bytes)
                S_c = r.range_decode(encoded, c_uniform, self.times).reshape(1, H, W)
                S_c = S_c.to(pe.DEVICE)
                S.append(S_c)

        S = torch.stack(S, dim=1)
        assert S.shape == (1, C, H, W)
        return S

    def _get_uniform_cdf(self, S_Shape, L):
        # TODO: could be cached, fixed, or done via an `arange`.
        with self.times.run('get_uniform_cdf'):
            return _get_cdf_from_pr(_get_uniform_pr(S_Shape, L))

    def encode_scale(self, scale, dmll, out, img, fout):
        """ Encode scale `scale`. """
        l = out.P[scale]
        bn = out.bn[scale] if scale != 0 else img
        S = out.S[scale]

        # shape used for all!
        write_shape(S.shape, fout)
        overhead_bytes = 5
        overhead_bytes += 4 * S.shape[1]

        r = ArithmeticCoder(dmll.L)

        # We encode channel by channel, because that's what's needed for the RGB scale. For s > 0, this could be done
        # in parallel for all channels
        def encoder(c, C_cur):
            S_c = S[:, c, ...].to(torch.int16)
            encoded = r.range_encode(S_c, cdf=C_cur, time_logger=self.times)
            write_num_bytes_encoded(len(encoded), fout)
            fout.write(encoded)
            # yielding always bottleneck and extra_info
            return bn[:, c, ...], len(encoded)

        with self.times.prefix_scope('encode scale'):
            with self.times.run('total'):
                _, entropy_coding_bytes_per_c = \
                    self.code_with_cdf(l, bn.shape, encoder, dmll)

        # --- cleanup
        out.P[scale] = None
        out.bn[scale] = None
        out.S[scale] = None
        # ---

        return sum(entropy_coding_bytes_per_c)

    def decode_scale(self, dmll, l, fin):
        C, H, W = read_shapes(fin)
        r = ArithmeticCoder(dmll.L)

        # We decode channel by channel, see `encode_scale`.
        def decoder(_, C_cur):
            num_bytes = read_num_bytes_encoded(fin)
            encoded = fin.read(num_bytes)
            S_c = r.range_decode(encoded, cdf=C_cur, time_logger=self.times).reshape(1, H, W)
            S_c = S_c.to(l.device, non_blocking=True)  # TODO: do directly in the extension
            bn_c = dmll.to_bn(S_c)
            # yielding always bottleneck and extra_info (=None here)
            return bn_c, None

        with self.times.prefix_scope('decode scale'):
            with self.times.run('total'):
                bn, _ = self.code_with_cdf(l, (1, C, H, W), decoder, dmll)

        return bn

    def code_with_cdf(self, l, bn_shape, bn_coder, dmll):
        """
        :param l: predicted distribution, i.e., NKpHW, see DiscretizedMixLogisticLoss
        :param bn_shape: shape of the bottleneck to encode/decode
        :param bn_coder: function with signature (c: int, C_cur: CDFOut) -> (bottleneck[c], extra_info_c). This is
        called for every channel of the bottleneck, with C_cur == CDF to use to encode/decode the channel. It shoud
        return the bottleneck[c].
        :param dmll: instance of DiscretizedMixLogisticLoss
        :return: decoded bottleneck, list of all extra info produced by `bn_coder`.
        """
        N, C, H, W = bn_shape
        coding = bitcoding.coders_helpers.CodingCDFNonshared(
                l, total_C=C, dmll=dmll)

        # needed also while encoding to get next C
        decoded_bn = torch.zeros(N, C, H, W, dtype=torch.float32).to(l.device)
        extra_info = []

        with self.times.combine('c{} {:.5f}'):
            for c in range(C):
                with self.times.run('get_C'):
                    C_cond_cur = coding.get_next_C(decoded_bn)
                with self.times.run('bn_coder'):
                    decoded_bn[:, c, ...], extra_info_c = bn_coder(c, C_cond_cur)
                    extra_info.append(extra_info_c)

        return decoded_bn, extra_info


def _get_cdf_from_pr(pr):
    """
    :param pr: NHWL
    :return: NHW(L+1) as int16 on CPU!
    """
    N, H, W, _ = pr.shape

    precision = 16

    cdf = torch.cumsum(pr, -1)
    cdf = cdf.mul_(2**precision)
    cdf = cdf.round()
    cdf = torch.cat((torch.zeros((N, H, W, 1), dtype=cdf.dtype, device=cdf.device),
                     cdf), dim=-1)
    cdf = cdf.to('cpu', dtype=torch.int16, non_blocking=True)

    return cdf


def _get_uniform_pr(S_shape, L):
    N, C, H, W = S_shape
    assert N == 1
    histo = torch.ones(L, dtype=torch.float32) / L
    assert (1 - histo.sum()).abs() < 1e-5, (1 - histo.sum()).abs()
    extendor = torch.ones(N, H, W, L)
    pr = extendor * histo
    return pr.to(pe.DEVICE)


def write_shape(shape, fout):
    """
    Write tuple (C,H,W) to file, given shape 1CHW.
    :return number of bytes written
    """
    assert len(shape) == 4 and shape[0] == 1, shape
    shape = shape[1:]
    assert shape[0] < 2**8,  shape
    assert shape[1] < 2**16, shape
    assert shape[2] < 2**16, shape
    assert len(shape) == 3,  shape
    write_bytes(fout, [np.uint8, np.uint16, np.uint16], shape)
    return 5


def read_shapes(fin):
    return tuple(map(int, read_bytes(fin, [np.uint8, np.uint16, np.uint16])))


def write_num_bytes_encoded(num_bytes, fout):
    assert num_bytes < 2**32
    write_bytes(fout, [np.uint32], [num_bytes])
    return 2  # number of bytes written


def read_num_bytes_encoded(fin):
    return int(read_bytes(fin, [np.uint32])[0])


def write_bytes(f, ts, xs):
    for t, x in zip(ts, xs):
        f.write(t(x).tobytes())


@ft.return_list
def read_bytes(f, ts):
    for t in ts:
        num_bytes_to_read = t().itemsize
        yield np.frombuffer(f.read(num_bytes_to_read), t, count=1)


# ---


def test_write_shapes(tmpdir):
    p = str(tmpdir.mkdir('test').join('hi.l3c'))
    with open(p, 'wb') as f:
        write_shape((1,2,3), f)
    with open(p, 'rb') as f:
        assert read_shapes(f) == (1,2,3)


def test_write_bytes(tmpdir):
    p = str(tmpdir.mkdir('test').join('hi.l3c'))
    with open(p, 'wb') as f:
        write_num_bytes_encoded(1234567, f)
    with open(p, 'rb') as f:
        assert read_num_bytes_encoded(f) == 1234567


def test_bytes(tmpdir):
    shape = (3, 512, 768)
    p = str(tmpdir.mkdir('test').join('hi.l3c'))
    with open(p, 'wb') as f:
        write_bytes(f, [np.uint8, np.uint16, np.uint16], shape)
    with open(p, 'rb') as f:
        c, h, w = read_bytes(f, [np.uint8, np.uint16, np.uint16])
        assert (c, h, w) == shape
