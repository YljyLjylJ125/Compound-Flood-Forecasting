import math

import torch

from anchored_forecaster.metrics.episodes import extract_episodes, match_episodes
from anchored_forecaster.metrics.forecast import peak_magnitude_error


def test_peak_error_ignores_invalid_values():
    pred = torch.tensor([[[1.0, 100.0, 3.0]]])
    target = torch.tensor([[[1.0, -100.0, 2.0]]])
    mask = torch.tensor([[[1, 0, 1]]], dtype=torch.bool)
    assert torch.isclose(peak_magnitude_error(pred, target, mask), torch.tensor(1.0))


def test_episode_merge_and_one_to_one_matching():
    values = torch.tensor([0.0, 2.0, 2.0, 0.0, 2.0, 0.0])
    events = extract_episodes(values, torch.ones_like(values).bool(), 1.0, min_duration=3, merge_gap=1)
    assert events == [{
        "start": 1,
        "end": 5,
        "duration": 4,
        "exceedance_duration": 3,
        "peak_index": 1,
        "peak_value": 2.0,
        "volume": 3.0,
    }]
    matches, misses, false_alarms = match_episodes(events, events + [{**events[0], "start": 5, "end": 6}])
    assert matches == [(0, 0)]
    assert misses == []
    assert false_alarms == [1]

