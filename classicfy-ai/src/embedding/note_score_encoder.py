"""Note features -> masked onset means -> BiGRU -> note-wise score context."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from features.note_score_context import (
    CATEGORICAL_FIELDS, FLAG_FIELDS, NUMERIC_FIELDS, PAD, UNK, VOCABULARIES,
)


@dataclass(frozen=True)
class NoteScoreEncoderConfig:
    embedding_dim: int = 8
    note_dim: int = 64
    hidden_size: int = 64
    context_dim: int = 128

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (self.embedding_dim, self.note_dim, self.hidden_size, self.context_dim)):
            raise ValueError('Encoder dimensions must be positive integers')


class NoteScoreEncoder(nn.Module):
    """Return [B,N,context_dim]; written tie continuations remain real notes."""

    def __init__(self, config: NoteScoreEncoderConfig | None = None):
        super().__init__()
        self.config = config or NoteScoreEncoderConfig()
        c = self.config
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(VOCABULARIES[name]), c.embedding_dim, padding_idx=PAD)
            for name in CATEGORICAL_FIELDS
        ])
        known_dim = len(NUMERIC_FIELDS + CATEGORICAL_FIELDS + FLAG_FIELDS)
        input_dim = len(NUMERIC_FIELDS) + len(FLAG_FIELDS) + len(CATEGORICAL_FIELDS) * c.embedding_dim + known_dim
        self.note_mlp = nn.Sequential(nn.Linear(input_dim, c.note_dim), nn.GELU(),
                                      nn.Linear(c.note_dim, c.note_dim))
        # Masked mean, log1p(note count), quarter gap, first-onset flag.
        self.gru = nn.GRU(c.note_dim + 3, c.hidden_size, batch_first=True, bidirectional=True)
        self.context_mlp = nn.Sequential(nn.Linear(c.note_dim + 2 * c.hidden_size, c.context_dim),
                                         nn.GELU(), nn.Linear(c.context_dim, c.context_dim))

    def forward(self, batch):
        numeric, categorical, flags = (batch[k] for k in ('numeric', 'categorical', 'flags'))
        known, notes, groups = (batch[k] for k in ('field_known', 'note_mask', 'onset_mask'))
        mapping, delta = batch['note_to_onset'], batch['delta_q']
        if numeric.ndim != 3 or min(numeric.shape[:2]) < 1:
            raise ValueError('Expected nonempty [B,N,Dnumeric]')
        b, n, _ = numeric.shape
        if groups.ndim != 2 or groups.shape[0] != b or groups.shape[1] < 1:
            raise ValueError('Expected nonempty [B,G] onset mask')
        g = groups.shape[1]
        shapes = ((numeric, (b, n, len(NUMERIC_FIELDS))),
                  (categorical, (b, n, len(CATEGORICAL_FIELDS))),
                  (flags, (b, n, len(FLAG_FIELDS))),
                  (known, (b, n, len(NUMERIC_FIELDS + CATEGORICAL_FIELDS + FLAG_FIELDS))),
                  (notes, (b, n)), (mapping, (b, n)), (delta, (b, g)))
        parameter = self.gru.weight_ih_l0
        if any(value.shape != shape for value, shape in shapes):
            raise ValueError('Note score batch shapes do not match the feature schema')
        if any(value.device != parameter.device for value, _ in shapes) or groups.device != parameter.device:
            raise ValueError('Batch and encoder must share a device')
        if any(value.dtype != parameter.dtype for value in (numeric, flags, delta)):
            raise ValueError('Floating input dtype must match the encoder')
        if any(value.dtype != torch.bool for value in (known, notes, groups)):
            raise ValueError('Feature-known and padding masks must be boolean')
        if categorical.dtype != torch.long or mapping.dtype != torch.long:
            raise ValueError('Categories and onset mapping must be int64')
        for mask in (notes, groups):
            lengths = mask.sum(dim=1)
            expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
            if torch.any(lengths == 0) or torch.any(mask != expected):
                raise ValueError('Each item needs real notes/onsets followed by right padding only')
        if torch.any(notes & ((mapping < 0) | (mapping >= g))):
            raise ValueError('Real notes need valid onset indices')
        safe_mapping = torch.where(notes, mapping, 0)
        if torch.any(notes & ~groups.gather(1, safe_mapping)):
            raise ValueError('A real note cannot reference a padded onset')
        if not torch.isfinite(delta[groups]).all() or torch.any(delta[groups] < 0):
            raise ValueError('Onset quarter gaps must be finite and nonnegative')
        if torch.any(delta[:, 0] != 0) or torch.any(groups[:, 1:] & (delta[:, 1:] <= 0)):
            raise ValueError('First onset gap must be zero; later real onset gaps must be positive')
        dn, dc = len(NUMERIC_FIELDS), len(CATEGORICAL_FIELDS)
        effective_known = known & notes.unsqueeze(-1)
        numeric_known, category_known, flag_known = torch.split(effective_known, (dn, dc, len(FLAG_FIELDS)), dim=-1)
        clean_numeric = torch.where(numeric_known, numeric, 0)
        clean_flags = torch.where(flag_known, flags, 0)
        if not torch.isfinite(clean_numeric).all() or not torch.isfinite(clean_flags).all():
            raise ValueError('Known real note features must be finite')
        if torch.any((clean_flags != 0) & (clean_flags != 1)):
            raise ValueError('Notation flags must be binary')
        encoded = []
        for j, embedding in enumerate(self.embeddings):
            ids = torch.where(category_known[:, :, j], categorical[:, :, j], UNK)
            ids = torch.where(notes, ids, PAD)
            if torch.any(notes & ((ids < 1) | (ids >= embedding.num_embeddings))):
                raise ValueError(f'Invalid category IDs for {CATEGORICAL_FIELDS[j]}')
            if torch.any(category_known[:, :, j] & (ids == UNK)):
                raise ValueError('UNK categories must have field_known=False')
            encoded.append(embedding(ids))
        inputs = torch.cat([clean_numeric, clean_flags, *encoded, effective_known.to(numeric.dtype)], dim=-1)
        h = self.note_mlp(inputs)
        h = torch.where(notes.unsqueeze(-1), h, 0)
        counts = numeric.new_zeros((b, g)).scatter_add(1, safe_mapping, notes.to(numeric.dtype))
        if torch.any(groups & (counts == 0)):
            raise ValueError('Each real onset group must contain at least one note')
        sums = numeric.new_zeros((b, g, self.config.note_dim)).scatter_add(
            1, safe_mapping.unsqueeze(-1).expand(-1, -1, self.config.note_dim), h)
        means = sums / counts.clamp_min(1).unsqueeze(-1)
        pooled = torch.cat([means, counts.log1p().unsqueeze(-1)], dim=-1)
        pooled = torch.where(groups.unsqueeze(-1), pooled, 0)
        first = numeric.new_zeros((b, g))
        first[:, 0] = 1
        gru_inputs = torch.cat([pooled, torch.where(groups, delta, 0).unsqueeze(-1), first.unsqueeze(-1)], dim=-1)
        packed = pack_padded_sequence(gru_inputs, groups.sum(dim=1).to('cpu'),
                                      batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        context, _ = pad_packed_sequence(output, batch_first=True, total_length=g)
        at_notes = context.gather(1, safe_mapping.unsqueeze(-1).expand(-1, -1, 2 * self.config.hidden_size))
        result = self.context_mlp(torch.cat([h, at_notes], dim=-1))
        return torch.where(notes.unsqueeze(-1), result, 0)
