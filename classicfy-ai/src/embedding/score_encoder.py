"""Position-wise score context for a conditional performance autoencoder."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


@dataclass(frozen=True)
class ScoreEncoderConfig:
    score_dim: int
    hidden_size: int = 64

    def __post_init__(self):
        if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (self.score_dim,self.hidden_size)):
            raise ValueError('Score dimensions must be positive integers')

    @property
    def context_dim(self):
        return 2*self.hidden_size


class ScoreEncoder(nn.Module):
    """[B,K,T,Ds] -> [B,K,T,2H], with zero output at all padding positions.

    Uses actual sequence lengths in both directions. Inactive segments never
    reach the GRU; flatten/unflatten preserves recording and segment ownership.
    """

    def __init__(self, config: ScoreEncoderConfig):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(config.score_dim,config.hidden_size,batch_first=True,bidirectional=True)

    def forward(self, score, time_mask, segment_mask):
        if score.ndim!=4 or score.shape[-1]!=self.config.score_dim or min(score.shape[:3])<1:
            raise ValueError('Expected nonempty [B,K,T,configured score_dim]')
        if score.dtype!=self.gru.weight_ih_l0.dtype or score.device!=self.gru.weight_ih_l0.device:
            raise ValueError('Score dtype/device must match the encoder')
        if (time_mask.shape!=score.shape[:3] or segment_mask.shape!=score.shape[:2]
                or time_mask.dtype!=torch.bool or segment_mask.dtype!=torch.bool
                or time_mask.device!=score.device or segment_mask.device!=score.device):
            raise ValueError('Expected matching boolean masks on the score device')
        if torch.any(~segment_mask.any(dim=1)):
            raise ValueError('Every recording needs a valid segment')
        valid = time_mask & segment_mask.unsqueeze(-1)
        lengths = valid.sum(dim=-1)
        if torch.any(segment_mask & (lengths==0)):
            raise ValueError('Valid segments need real time steps')
        expected = torch.arange(score.shape[2],device=score.device)<lengths.unsqueeze(-1)
        if torch.any(valid!=expected):
            raise ValueError('Real time steps must be a contiguous prefix; right padding only')
        if not torch.isfinite(score[valid]).all():
            raise ValueError('Score features at real time steps must be finite')
        clean = torch.where(valid.unsqueeze(-1),score,0)
        b,k,t,ds = score.shape
        active = torch.nonzero(segment_mask.reshape(-1),as_tuple=False).flatten()
        packed = pack_padded_sequence(clean.reshape(b*k,t,ds)[active],
            lengths.reshape(-1)[active].to('cpu'),batch_first=True,enforce_sorted=False)
        encoded,_ = self.gru(packed)
        dense,_ = pad_packed_sequence(encoded,batch_first=True,total_length=t)
        flat = score.new_zeros((b*k,t,self.config.context_dim)).index_copy(0,active,dense)
        return flat.reshape(b,k,t,self.config.context_dim)
