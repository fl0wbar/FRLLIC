import torch

from criterion.logistic_mixture import DiscretizedMixLogisticLoss, CDFOut


class CodingCDFNonshared(object):
    def __init__(self, l, total_C, dmll: DiscretizedMixLogisticLoss):
        """
        :param l: predicted distribution, i.e., NKpHW, see DiscretizedMixLogisticLoss
        :param total_C:
        :param dmll:
        """
        self.l = l
        self.dmll = dmll

        # Lp = L+1
        self.targets = torch.linspace(dmll.x_min - dmll.bin_width / 2,
                                      dmll.x_max + dmll.bin_width / 2,
                                      dmll.L + 1, dtype=torch.float32, device=l.device)
        self.total_C = total_C
        self.c_cur = 0

    def get_next_C(self, decoded_x) -> CDFOut:
        """
        Get CDF to encode/decode next channel
        :param decoded_x: NCHW
        :return: C_cond_cur, NHWL'
        """
        C_Cur = self.dmll.cdf_step_non_shared(self.l, self.targets, self.c_cur, self.total_C, decoded_x)
        self.c_cur += 1
        return C_Cur

