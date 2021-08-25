import torch.nn.functional as F
from .transformer import Transformer, ActivationFunction
from .universal_transformer import UniversalTransformerDecoderWithLayer, UniversalTransformerEncoderWithLayer
from .relative_transformer import RelativeTransformerDecoderLayer, RelativeTransformerEncoderLayer


class UniversalRelativeTransformer(Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: ActivationFunction = F.relu):

        super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation,
                         UniversalTransformerEncoderWithLayer(RelativeTransformerEncoderLayer),
                         UniversalTransformerDecoderWithLayer(RelativeTransformerDecoderLayer))
