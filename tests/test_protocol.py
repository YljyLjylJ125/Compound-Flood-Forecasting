from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_paper import (  # noqa: E402
    ARCHITECTURE_ABLATIONS,
    PAPER_MODELS,
    SOURCE_ABLATIONS,
    paper_matrix,
)


def test_paper_matrix_matches_manuscript_surface():
    runs = paper_matrix()
    assert len(runs) == 1062
    assert sum(run.suite == "main" for run in runs) == 972
    assert sum(run.suite == "architecture_ablation" for run in runs) == 45
    assert sum(run.suite == "source_ablation" for run in runs) == 45

    assert {run.model for run in runs if run.suite == "main"} == set(PAPER_MODELS)
    assert {run.model for run in runs if run.suite == "architecture_ablation"} == set(ARCHITECTURE_ABLATIONS)
    assert {run.model for run in runs if run.suite == "source_ablation"} == set(SOURCE_ABLATIONS)
    assert len({run.relative_dir for run in runs}) == len(runs)

    for run in runs:
        if run.suite != "main":
            assert run.split == "S_7"
            assert run.horizon == "3D"


def test_restricted_matrix_preserves_requested_parts_and_repeats():
    runs = paper_matrix(splits=("S_6",), parts=(2,), repeat_ids=(3,))
    assert len(runs) == len(PAPER_MODELS) * 4
    assert {run.split for run in runs} == {"S_6"}
    assert {run.part for run in runs} == {2}
    assert {run.repeat_id for run in runs} == {3}
