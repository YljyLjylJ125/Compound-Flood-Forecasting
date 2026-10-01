from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from anchored_forecaster.evaluation import save_prediction_archive


def test_prediction_archive_uses_physical_daily_issue_grid(tmp_path):
    samples = 49
    prediction = torch.arange(samples, dtype=torch.float32).view(samples, 1, 1)
    target = prediction + 1
    mask = torch.ones_like(prediction)
    dataset = SimpleNamespace(
        lookback=48,
        timestamps=list(pd.date_range("2023-01-01", periods=200, freq="h")),
    )
    path = tmp_path / "predictions.npz"
    save_prediction_archive(path, prediction, target, mask, dataset, issue_stride=24)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["issue_index"].tolist() == [0, 24, 48]
        assert archive["prediction"][:, 0, 0].tolist() == [0.0, 24.0, 48.0]
        assert int(archive["archive_issue_stride_h"]) == 24
