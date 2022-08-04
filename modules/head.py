from torch import nn

from modules import edsr
from pytorch_ext import default_conv as conv



class RGBHead(nn.Module):
    """ Go from 3 channels (RGB) to Cf channels, also normalize RGB """
    def __init__(self, config_ms):
        super(RGBHead, self).__init__()
        assert 'Subsampling' not in config_ms.enc.cls, 'For Subsampling encoders, head should be ID'
        self.head = nn.Sequential(
                edsr.MeanShift(0, (0., 0., 0.), (128., 128., 128.)),
                Head(config_ms, Cin=3))
        self._repr = 'MeanShift//Head(C=3)'

    def __repr__(self):
        return f'RGBHead({self._repr})'

    def forward(self, x):
        return self.head(x)


class Head(nn.Module):
    """
    Go from Cin channels to Cf channels.
    For L3C, Cin=Cf, and this is the convolution yielding E^{s+1}_in in Fig. 2.

    """
    def __init__(self, config_ms, Cin):
        super(Head, self).__init__()
        assert 'Subsampling' not in config_ms.enc.cls, 'For Subsampling encoders, head should be ID'
        self.head = conv(Cin, config_ms.Cf, config_ms.kernel_size)
        self._repr = f'Conv({config_ms.Cf})'

    def __repr__(self):
        return f'Head({self._repr})'

    def forward(self, x):
        return self.head(x)


