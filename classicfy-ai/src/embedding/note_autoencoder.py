"""Conditional, non-autoregressive note sequence AE; tempo is shared per beat."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .note_score_encoder import NoteScoreEncoder, NoteScoreEncoderConfig


@dataclass(frozen=True)
class NoteAEConfig:
    context_dim: int = 128
    condition_dim: int = 64
    input_dim: int = 128
    hidden_size: int = 64
    layers: int = 2
    dropout: float = .1
    latent_dim: int = 64
    score_gradient: str = 'joint'
    mean_tempo_auxiliary: bool = False

    def __post_init__(self):
        for name in ('context_dim', 'condition_dim', 'input_dim', 'hidden_size', 'layers', 'latent_dim'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('AE dimensions must be positive integers')
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1 or self.score_gradient not in {'joint', 'frozen'}:
            raise ValueError('Invalid AE dropout or score gradient policy')
        if not isinstance(self.mean_tempo_auxiliary, bool):
            raise ValueError('mean_tempo_auxiliary must be boolean')


def validate_padding(P, lengths):
    if P.ndim != 2 or min(P.shape) < 1 or P.dtype != torch.bool:
        raise ValueError('P must be nonempty boolean [B,N]')
    if lengths.shape != P.shape[:1] or lengths.dtype != torch.long:
        raise ValueError('lengths must be int64 [B]')
    expected = torch.arange(P.shape[1], device=P.device)[None] < lengths.to(P.device)[:, None]
    if torch.any(lengths <= 0) or torch.any(lengths > P.shape[1]) or not torch.equal(expected, P):
        raise ValueError('Positive lengths and contiguous right padding required')


def _packed(gru, inputs, lengths):
    packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
    result, _ = gru(packed)
    return pad_packed_sequence(result, batch_first=True, total_length=inputs.shape[1])[0]


class ConditionalNoteAutoencoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or NoteAEConfig()
        c = self.config
        self.condition = nn.Linear(c.context_dim, c.condition_dim)
        self.note_mlp = nn.Sequential(nn.Linear(c.condition_dim + 22, c.input_dim), nn.GELU(),
                                      nn.Linear(c.input_dim, c.input_dim))
        opts = dict(hidden_size=c.hidden_size, num_layers=c.layers, batch_first=True,
                    bidirectional=True, dropout=c.dropout if c.layers > 1 else 0.)
        self.encoder = nn.GRU(c.input_dim, **opts)
        self.latent = nn.Sequential(nn.Linear(2 * c.hidden_size, c.latent_dim), nn.GELU(),
                                    nn.Linear(c.latent_dim, c.latent_dim))
        self.decoder = nn.GRU(c.condition_dim + c.latent_dim, **opts)
        self.onset_head = nn.Linear(2 * c.hidden_size, 1)
        self.velocity_head = nn.Linear(2 * c.hidden_size, 1)
        self.articulation_head = nn.Linear(2 * c.hidden_size, 1)
        self.pedal_head = nn.Linear(2 * c.hidden_size, 7)
        self.tempo_head = nn.Linear(2 * c.hidden_size, 1)
        self.mean_tempo_head = None
        if c.mean_tempo_auxiliary:
            # Adding this deterministic head must not change shared initialization
            # or the seeded dropout stream in a controlled baseline comparison.
            with torch.random.fork_rng(devices=[]):
                self.mean_tempo_head = nn.Sequential(nn.Linear(c.latent_dim, c.latent_dim), nn.GELU(),
                                                      nn.Linear(c.latent_dim, 1))

    def _condition(self, C, P, lengths):
        validate_padding(P, lengths)
        if C.shape != (*P.shape, self.config.context_dim):
            raise ValueError('C shape must match notes and configured context dimension')
        clean = torch.where(P[..., None], C, 0.)
        if not torch.isfinite(clean).all():
            raise ValueError('Observed C must be finite')
        return torch.where(P[..., None], self.condition(clean), 0.)

    def encode(self, C, R, M, P, lengths):
        condition = self._condition(C, P, lengths)
        if R.shape != (*P.shape, 11) or M.shape != R.shape or M.dtype != torch.bool:
            raise ValueError('Expected R and boolean M [B,N,11]')
        observed = M & P[..., None]
        clean = torch.where(observed, R, 0.)
        if not torch.isfinite(clean).all():
            raise ValueError('Observed R must be finite')
        x = self.note_mlp(torch.cat([condition, clean, observed.to(clean.dtype)], dim=-1))
        h = _packed(self.encoder, x, lengths)
        pooled = torch.where(P[..., None], h, 0.).sum(1) / lengths.to(h.device)[:, None]
        z = self.latent(pooled)
        result = {'z': z, 'e': F.normalize(z, dim=-1)}
        if self.mean_tempo_head is not None:
            result['mean_tempo_hat'] = self.mean_tempo_head(z).squeeze(-1)
        return result

    def decode(self, C, z, P, lengths, beat_id):
        condition = self._condition(C, P, lengths)
        if z.shape != (P.shape[0], self.config.latent_dim) or not torch.isfinite(z).all():
            raise ValueError('Expected finite, unnormalized z [B,d_z]')
        if beat_id.shape != P.shape or beat_id.dtype != torch.long or torch.any(P & (beat_id < -1)):
            raise ValueError('Expected beat IDs [B,N], with -1 for unavailable correspondence')
        # Only score context and the shared latent enter the decoder.
        d = _packed(self.decoder, torch.cat([condition, z[:, None].expand(-1, P.shape[1], -1)], -1), lengths)
        keys = [torch.unique(beat_id[i, P[i] & (beat_id[i] >= 0)], sorted=True) for i in range(len(P))]
        width = max((len(k) for k in keys), default=0)
        beat_values = d.new_zeros((len(P), width))
        beat_keys = beat_id.new_full((len(P), width), -1)
        beat_mask = torch.zeros_like(beat_keys, dtype=torch.bool)
        tempo = d.new_zeros((*P.shape, 1))
        for i, ids in enumerate(keys):
            for j, key in enumerate(ids):
                rows = P[i] & (beat_id[i] == key)
                value = self.tempo_head(d[i, rows].mean(0)).squeeze(-1)
                beat_values[i, j], beat_keys[i, j], beat_mask[i, j] = value, key, True
                tempo[i, rows, 0] = value
        prediction = torch.cat([tempo, self.onset_head(d), self.velocity_head(d),
                                self.articulation_head(d), self.pedal_head(d)], -1)
        return {'R_hat': torch.where(P[..., None], prediction, 0.), 'beat_prediction': beat_values,
                'beat_keys': beat_keys, 'beat_mask': beat_mask}

    def forward(self, C, R, M, P, lengths, beat_id):
        encoded = self.encode(C, R, M, P, lengths)
        return {**encoded, **self.decode(C, encoded['z'], P, lengths, beat_id)}


class NotePerformanceAutoencoder(nn.Module):
    """Existing ScoreEncoder produces C on demand; gradients are explicit in config."""
    def __init__(self, config=None, score_config=None):
        super().__init__()
        self.config = config or NoteAEConfig()
        self.score_encoder = NoteScoreEncoder(score_config or NoteScoreEncoderConfig(context_dim=self.config.context_dim))
        if self.score_encoder.config.context_dim != self.config.context_dim:
            raise ValueError('Score context dimension differs from AE context dimension')
        self.ae = ConditionalNoteAutoencoder(self.config)
        self.r_scaler_fingerprint = None
        self.score_scale_fingerprint = None
        if self.config.score_gradient == 'frozen':
            self.score_encoder.requires_grad_(False)
            self.score_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.config.score_gradient == 'frozen':
            self.score_encoder.eval()
        return self

    def context(self, batch):
        if batch.get('R_state') != 'train-channel-scaled-once':
            raise ValueError('AE requires exactly-once scaled raw R')
        if self.r_scaler_fingerprint is not None and batch['R_scaler']['fingerprint'] != self.r_scaler_fingerprint:
            raise ValueError('AE R scaler mismatch')
        if self.score_scale_fingerprint is not None and batch.get('score_scale_id') != self.score_scale_fingerprint:
            raise ValueError('AE score scale mismatch')
        if self.config.score_gradient == 'frozen':
            with torch.no_grad():
                return self.score_encoder(batch)
        return self.score_encoder(batch)

    def encode_batch(self, batch):
        return self.ae.encode(self.context(batch), batch['R'], batch['M'], batch['P'], batch['lengths'])

    def forward(self, batch):
        return self.ae(self.context(batch), batch['R'], batch['M'], batch['P'], batch['lengths'], batch['beat_id'])
