import torch
from torchac import torchac
from common import no_op


from criterion.logistic_mixture import CDFOut
from test.cuda_timer import StackTimeLogger


class ArithmeticCoder(object):
    def __init__(self, L):
        self.L = L
        self._cached_cdf = None

    def range_encode(self, data, cdf, time_logger: StackTimeLogger):
        """
        :param data: data to encode
        :param cdf: cdf to use, either a NHWLp matrix or instance of CDFOut
        :return: data encode to a bytes string
        """
        assert len(data.shape) == 3, data.shape

        with time_logger.run('data -> cpu'):
            data = data.to('cpu', non_blocking=True)
        assert data.dtype == torch.int16, 'Wrong dtype: {}'.format(data.dtype)

        with time_logger.run('reshape'):
            data = data.reshape(-1).contiguous()

        if isinstance(cdf, CDFOut):
            logit_probs_c_sm, means_c, log_scales_c, K, targets = cdf

            with time_logger.run('ac.encode'):
                out_bytes = torchac.encode_logistic_mixture(
                        targets, means_c, log_scales_c, logit_probs_c_sm, data)
        else:
            N, H, W, Lp = cdf.shape
            assert Lp == self.L + 1, (Lp, self.L)

            with time_logger.run('ac.encode'):
                out_bytes = torchac.encode_cdf(cdf, data)

        return out_bytes

    def range_decode(self, encoded_bytes, cdf, time_logger: StackTimeLogger = no_op.NoOp):
        """
        :param encoded_bytes: bytes encoded by range_encode
        :param cdf: cdf to use, either a NHWLp matrix or instance of CDFOut
        :return: decoded matrix as np.int16, NHW
        """
        if isinstance(cdf, CDFOut):
            logit_probs_c_sm, means_c, log_scales_c, K, targets = cdf

            N, _, H, W = means_c.shape

            with time_logger.run('ac.encode'):
                decoded = torchac.decode_logistic_mixture(
                        targets, means_c, log_scales_c, logit_probs_c_sm, encoded_bytes)

        else:
            N, H, W, Lp = cdf.shape
            assert Lp == self.L + 1, (Lp, self.L)

            with time_logger.run('ac.encode'):
                decoded = torchac.decode_cdf(cdf, encoded_bytes)

        return decoded.reshape(N, H, W)
