
from core.models.laplace import Laplacian
from core.models.wavelets import ComplexDaubechiesWavelets
from core.models.models import MultiResolutionFusion
from core.models.models import FusionNonLinearWeightedAverageModel
from core.models.unet import UNet
from core.models.unet import AttentionUNet
from core.models.ebsr import EBSR

models_config = {
  'laplacian': Laplacian,
  'wavelets': ComplexDaubechiesWavelets,
  'softmax_average': FusionNonLinearWeightedAverageModel,
  'fusion': MultiResolutionFusion,
  'unet': UNet,
  'attention_unet': AttentionUNet,
  'ebsr': EBSR,
}
