"""연주 해석 feature의 공개 인터페이스."""

from .articulation import (
    ArticulationFeature,
    NoteArticulation,
    build_tempo_map,
    extract_articulation,
    extract_note_articulation,
    summarize_articulation,
)
from .beat_grid import BeatGrid, as_beat_array, assign_windows, build_beat_grid
from .common_pattern import PieceFeatureInput, SeparatedFeature, separate_piece_feature
from .dynamics import DynamicsFeature, extract_dynamics, summarize_dynamics
from .models import BeatSequence
from .note_score_context import (
    NoteScoreFeatures, NoteScoreScale, encode_note_numeric, extract_note_score_context,
    fit_note_score_scale, note_score_schema,
)
from .score_context import (
    SCORE_CHANNELS, ScoreContextFeatures, ScoreContextScale, encode_score_features,
    extract_score_context, fit_score_context_scale, load_score_context, score_context_schema,
)
from .normalization import (
    DEFAULT_SCALE_METHODS, NormalizedFeature, ResidualScale, fit_residual_scale,
    standardize_piece_feature, standardize_sequence,
)
from .pedaling import PedalingFeature, extract_pedaling, summarize_pedaling
from .rubato import RubatoFeature, extract_piece_rubato_features, summarize_rubato
from .tempo import (
    TempoFeature,
    TempoInput,
    TempoInterval,
    extract_piece_tempo_features,
    summarize_tempo,
)

__all__ = [
    "ArticulationFeature",
    "BeatGrid",
    "BeatSequence",
    "NoteScoreFeatures",
    "NoteScoreScale",
    "encode_note_numeric",
    "extract_note_score_context",
    "fit_note_score_scale",
    "note_score_schema",
    "DynamicsFeature",
    "DEFAULT_SCALE_METHODS",
    "NormalizedFeature",
    "NoteArticulation",
    "PedalingFeature",
    "PieceFeatureInput",
    "RubatoFeature",
    "ResidualScale",
    "SeparatedFeature",
    "TempoFeature",
    "TempoInput",
    "TempoInterval",
    "as_beat_array",
    "assign_windows",
    "build_beat_grid",
    "build_tempo_map",
    "extract_articulation",
    "extract_dynamics",
    "extract_note_articulation",
    "extract_pedaling",
    "extract_piece_rubato_features",
    "extract_piece_tempo_features",
    "fit_residual_scale",
    "separate_piece_feature",
    "standardize_piece_feature",
    "standardize_sequence",
    "summarize_articulation",
    "summarize_dynamics",
    "summarize_pedaling",
    "summarize_rubato",
    "summarize_tempo",
    "SCORE_CHANNELS",
    "ScoreContextFeatures",
    "ScoreContextScale",
    "encode_score_features",
    "extract_score_context",
    "fit_score_context_scale",
    "load_score_context",
    "score_context_schema",
]
