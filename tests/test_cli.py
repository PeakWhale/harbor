import json

import pandas as pd

from conftest import EXAMPLE_CSV, TARGET
from peakwhale_harbor.cli import main


def test_train_command(tmp_path, capsys):
    out = tmp_path / "cli-bundle"
    code = main(["train", str(EXAMPLE_CSV), "--target", TARGET, "--out", str(out), "--drop", "site"])
    assert code == 0
    text = capsys.readouterr().out
    assert "Hold-out test results" in text
    assert "Typical error (MAE)" in text
    assert "skip   site" in text
    assert (out / "model.skops").is_file()


def test_train_command_errors(tmp_path, capsys):
    code = main(["train", str(EXAMPLE_CSV), "--target", "nope", "--out", str(tmp_path / "x")])
    assert code == 1
    assert "not in the file" in capsys.readouterr().err


def test_predict_command(bundle_dir, tmp_path):
    src = tmp_path / "in.csv"
    pd.read_csv(EXAMPLE_CSV).head(10).to_csv(src, index=False)
    dst = tmp_path / "out.csv"
    assert main(["predict", str(bundle_dir), str(src), "--out", str(dst)]) == 0
    out = pd.read_csv(dst)
    assert len(out) == 10
    assert (out[f"{TARGET}_low"] <= out[f"{TARGET}_prediction"]).all()


def test_predict_command_bad_input(bundle_dir, tmp_path, capsys):
    src = tmp_path / "in.csv"
    pd.read_csv(EXAMPLE_CSV).head(3).drop(columns=["shift"]).to_csv(src, index=False)
    assert main(["predict", str(bundle_dir), str(src)]) == 1
    assert "shift" in capsys.readouterr().err


def test_info_command(bundle_dir, capsys):
    assert main(["info", str(bundle_dir)]) == 0
    text = capsys.readouterr().out
    assert "machine_type" in text and "(not used) asset_id" in text
    assert main(["info", str(bundle_dir), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["target"]["name"] == TARGET


def test_missing_bundle(tmp_path, capsys):
    assert main(["info", str(tmp_path / "none")]) == 1
    assert "not found" in capsys.readouterr().err
