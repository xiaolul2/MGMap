from .vovnet import VoVNet

from .efficientnet.model import EfficientNetBackbone
from .swin_transformer.model import SwinTRBackbone
#from .resnet.model  import ResNetBackbone
from .swin import SwinTransformer
from .bifpn.model import BiFPN
from .resnet.model import ResNetBackbone
__all__ = ['VoVNet', 'EfficientNetBackbone','SwinTRBackbone','ResNetBackbone']