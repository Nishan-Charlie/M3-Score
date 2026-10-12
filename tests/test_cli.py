import json
import subprocess
import sys

import pytest

from _images import write_pngs
from m3score import M3Score, M3ScoreError, __version__, metric
from m3score.cli import main


@pytest.fixture()
def folders(tmp_path):
    write_pngs(tmp_path / "real", 14, seed=0)
    write_pngs(tmp_path / "gen", 12, seed=1)
    return str(tmp_path / "real"), str(tmp_path / "gen")


def test_prints_the_three_axes(offline_backbone, folders, capsys):
    assert main([*folders, "--device", "cpu", "--quiet"]) == 0
    out = capsys.readouterr().out
    for word in ("M3-Score", "m3-v1", "fidelity", "memorization", "coverage", "14 images"):
        assert word in out
    assert out.isascii()


def test_writes_json(offline_backbone, folders, tmp_path):
    target = tmp_path / "result.json"
    status = main([*folders, "--device", "cpu", "--quiet", "--no-stats", "--json", str(target)])
    assert status == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["m3score_version"] == __version__
    assert payload["n_real"] == 14
    assert payload["fidelity_ci"] is None


def test_n_is_passed_through(offline_backbone, folders, capsys):
    assert main([*folders, "--device", "cpu", "--quiet", "--no-stats", "--n", "8"]) == 0
    assert "8 images" in capsys.readouterr().out


@pytest.mark.filterwarnings("ignore::m3score.M3ScoreWarning")
def test_command_line_and_python_api_agree(offline_backbone, folders, tmp_path):
    target = tmp_path / "result.json"
    arguments = [*folders, "--device", "cpu", "--quiet", "--seed", "5", "--n", "10"]
    assert main([*arguments, "--json", str(target)]) == 0
    from_cli = json.loads(target.read_text(encoding="utf-8"))
    from_python = M3Score(device="cpu", seed=5).score(*folders, n=10).to_dict()
    assert {key: from_cli[key] for key in from_python} == from_python


def test_input_errors_exit_2_with_one_line(offline_backbone, tmp_path, capsys):
    absent = str(tmp_path / "absent")
    assert main([absent, absent, "--device", "cpu", "--quiet"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("m3score: error: ")
    assert err.count("\n") == 1


def test_backbone_failure_exits_1_with_one_line(monkeypatch, folders, capsys):
    def refuse(spec=None, device="cpu"):
        raise M3ScoreError("could not load the M3-Score backbone")

    monkeypatch.setattr(metric, "load_backbone", refuse)
    assert main([*folders, "--device", "cpu", "--quiet"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("m3score: error: could not load")
    assert err.count("\n") == 1


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_module_entry_point():
    done = subprocess.run(
        [sys.executable, "-m", "m3score", "--version"], capture_output=True, text=True
    )
    assert done.returncode == 0
    assert __version__ in done.stdout
