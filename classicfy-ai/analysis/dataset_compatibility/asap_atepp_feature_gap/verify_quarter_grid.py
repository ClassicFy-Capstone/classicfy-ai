"""Verify musical-unit conversion, tempo changes and special-interval masks."""
import tempfile
from pathlib import Path

import mido
import numpy as np

from quarter_grid import score_tempo_map, seconds_to_quarters, quarter_boundaries


with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "tempo.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([mido.MetaMessage("set_tempo", tempo=500000, time=0),
                  mido.MetaMessage("set_tempo", tempo=1000000, time=960),
                  mido.MetaMessage("end_of_track", time=960)])
    midi.save(path)
    q = seconds_to_quarters([0, .5, 1, 2, 3], score_tempo_map(str(path)))
    np.testing.assert_allclose(q, [0, 1, 2, 3, 4], rtol=0, atol=1e-12)
grid, times, valid, exact = quarter_boundaries([0, 2, 4], [1, 3, 7], ["b", "b", "b"])
np.testing.assert_array_equal(grid, [0, 1, 2, 3, 4])
np.testing.assert_allclose(times, [1, 2, 3, 5, 7], rtol=0, atol=1e-12)
np.testing.assert_array_equal(exact, [True, False, True, False, True])
assert valid.all()
grid, times, valid, exact = quarter_boundaries([0, 1, 2, 3], [0, 1, 2, 3], ["b", "bR", "b", "b"])
np.testing.assert_array_equal(valid, [False, False, True])
grid, times, valid, exact = quarter_boundaries([.5, 2, 3.5], [1, 4, 7], ["b", "b", "b"])
np.testing.assert_array_equal(grid, [1, 2, 3])
np.testing.assert_allclose(times, [2, 4, 6], rtol=0, atol=1e-12)
for q, t in [([0, 0, 1], [0, 1, 2]), ([0, 1, 2], [0, 1, 1])]:
    try:
        quarter_boundaries(q, t, ["b"] * len(q))
    except ValueError:
        pass
    else:
        raise AssertionError("Non-increasing anchors must be rejected")
print("PASS: tempo-map inversion, half-note subdivision, bR masking, no extrapolation, invalid-anchor rejection")
