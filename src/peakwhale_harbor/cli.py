"""The ``harbor`` command.

    harbor train data.csv --target price      train and save a bundle
    harbor serve models/data                  web form + JSON API
    harbor predict models/data new.csv        predictions for a CSV file
    harbor info models/data                   what the model expects and how good it is
    harbor mcp models/data                    MCP server for AI assistants
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .bundle import BundleError, load_bundle, read_csv_for_prediction
from .formatting import format_number, format_percent
from .schema import InputError
from .train import DEFAULT_COVERAGE, DEFAULT_SEED, TrainingError, train


def _names(values: Sequence[str] | None) -> list[str]:
    """Accept --drop a,b and --drop a --drop b."""
    out: list[str] = []
    for value in values or []:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "model"


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _print_problems(exc: InputError) -> None:
    print(f"error: {exc.message}", file=sys.stderr)
    for p in exc.problems:
        where = ", ".join(f"{k} {p[k]}" for k in ("line", "row", "field") if k in p)
        fields = f": {', '.join(p['fields'])}" if "fields" in p else ""
        print(f"  {where + ': ' if where else ''}{p['error']}{fields}", file=sys.stderr)


def print_metrics(metrics: dict, coverage: float) -> None:
    rows = [
        ("Typical error (MAE)", format_number(metrics["mae"])),
        ("Root mean squared error (RMSE)", format_number(metrics["rmse"])),
        ("R² (1.0 is perfect)", f"{metrics['r2']:.3f}"),
        ("Simple guess error (median)", format_number(metrics["baseline_mae"])),
        (
            f"{round(coverage * 100)}% range: true value inside",
            f"{format_percent(metrics['range_coverage'])} of test rows",
        ),
        ("Average range width", format_number(metrics["range_mean_width"])),
    ]
    print(f"Hold-out test results ({metrics['test_rows']:,} rows the model never saw):")
    for label, value in rows:
        print(f"  {label:<34} {value}")


def cmd_train(args: argparse.Namespace) -> int:
    descriptions = {}
    if args.descriptions:
        try:
            descriptions = json.loads(Path(args.descriptions).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _error(f"cannot read --descriptions file: {exc}")
        if not isinstance(descriptions, dict):
            return _error("--descriptions must be a JSON object: {\"column\": \"description\"}")
    out = Path(args.out) if args.out else Path("models") / _slug(Path(args.csv).name.split(".")[0])
    print(f"Training on {args.csv}, target column: {args.target}")
    try:
        bundle = train(
            args.csv,
            args.target,
            out,
            coverage=args.coverage,
            test_size=args.test_size,
            calibration_size=args.calibration_size,
            seed=args.seed,
            drop=_names(args.drop),
            categorical=_names(args.categorical),
            numeric=_names(args.numeric),
            title=args.title,
            description=args.description,
            descriptions=descriptions,
            log=print,
        )
    except (TrainingError, BundleError) as exc:
        return _error(str(exc))
    print()
    print_metrics(bundle.metrics, bundle.coverage)
    print()
    print(f"Saved bundle to {out}")
    print(f"Next: harbor serve {out}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import create_app

    try:
        app = create_app(args.bundle)
    except BundleError as exc:
        return _error(str(exc))
    print(f"Serving {args.bundle} on http://{args.host}:{args.port}  (press Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    try:
        bundle = load_bundle(args.bundle)
        df = read_csv_for_prediction(args.csv)
        out = bundle.predict_table(df)
    except BundleError as exc:
        return _error(str(exc))
    except InputError as exc:
        _print_problems(exc)
        return 1
    except (OSError, ValueError) as exc:
        return _error(f"cannot read {args.csv}: {exc}")
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"Wrote {len(out)} predictions to {args.out}")
    else:
        try:
            out.to_csv(sys.stdout, index=False)
        except BrokenPipeError:
            # The output went to a program that stopped reading early (like "head").
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    try:
        bundle = load_bundle(args.bundle)
    except BundleError as exc:
        return _error(str(exc))
    info = bundle.describe()
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0
    target = bundle.target
    print(f"{bundle.title}")
    print(f"Predicts: {target['name']}  {target.get('description', '')}".rstrip())
    print(f"Trained: {info['trained']['created_at']} from {info['trained']['source_file']}")
    print("Inputs:")
    for f in bundle.features:
        if f["type"] == "numeric":
            detail = f"number, {format_number(f['min'])} to {format_number(f['max'])}"
        else:
            detail = f"one of {len(f['categories'])} values: {', '.join(f['categories'][:6])}"
            if len(f["categories"]) > 6:
                detail += ", ..."
        print(f"  {f['name']:<28} {detail}")
    for d in info["dropped_columns"]:
        print(f"  (not used) {d['name']}: {d['reason']}")
    print()
    print_metrics(bundle.metrics, bundle.coverage)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    try:
        from .mcp_server import run_stdio
    except ImportError:
        return _error("the MCP server needs the 'mcp' package: pip install 'peakwhale-harbor[mcp]'")
    try:
        run_stdio(args.bundle)
    except BundleError as exc:
        return _error(str(exc))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harbor",
        description="PeakWhale Harbor: turn a CSV file into a prediction API with prediction ranges.",
    )
    parser.add_argument("--version", action="version", version=f"harbor {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    t = sub.add_parser("train", help="train a model bundle from a CSV file")
    t.add_argument("csv", help="CSV file with a header row")
    t.add_argument("--target", required=True, help="column to predict (must contain numbers)")
    t.add_argument("--out", help="bundle folder to write (default: models/<file name>)")
    t.add_argument(
        "--coverage", type=float, default=DEFAULT_COVERAGE, help="share of true values the range should hold (0.9)"
    )
    t.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed for the split and the models (42)")
    t.add_argument("--test-size", type=float, default=0.2, help="share of rows kept for testing (0.2)")
    t.add_argument(
        "--calibration-size", type=float, default=0.2, help="share of rows used to calibrate the range (0.2)"
    )
    t.add_argument("--drop", action="append", help="columns to ignore, comma separated")
    t.add_argument("--categorical", action="append", help="columns to treat as categories, comma separated")
    t.add_argument("--numeric", action="append", help="columns to treat as numbers, comma separated")
    t.add_argument("--title", help="name shown in the web page (default: from the file name)")
    t.add_argument("--description", default="", help="one sentence shown under the title")
    t.add_argument("--descriptions", help="JSON file with short descriptions: {\"column\": \"text\"}")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("serve", help="start the web form and JSON API for a bundle")
    s.add_argument("bundle", help="bundle folder")
    s.add_argument("--host", default="127.0.0.1", help="address to listen on (127.0.0.1)")
    s.add_argument("--port", type=int, default=5050, help="port (5050)")
    s.set_defaults(func=cmd_serve)

    p = sub.add_parser("predict", help="predict every row of a CSV file")
    p.add_argument("bundle", help="bundle folder")
    p.add_argument("csv", help="CSV file with the input columns")
    p.add_argument("--out", help="output CSV (default: print to the screen)")
    p.set_defaults(func=cmd_predict)

    i = sub.add_parser("info", help="show the inputs and test results of a bundle")
    i.add_argument("bundle", help="bundle folder")
    i.add_argument("--json", action="store_true", help="print JSON")
    i.set_defaults(func=cmd_info)

    m = sub.add_parser("mcp", help="run an MCP server (stdio) so AI assistants can use the bundle")
    m.add_argument("bundle", help="bundle folder")
    m.set_defaults(func=cmd_mcp)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
