"""Read CSV files the way spreadsheet programs write them.

* Separator: a comma, or a tab for .tsv files. Excel in many European
  languages writes a semicolon instead; Harbor finds it in the header row.
* Encoding: UTF-8. Files that are not UTF-8 (for example "CSV (Comma
  delimited)" from older Excel on Windows) are read as Windows-1252, and
  Harbor says so.
"""

from __future__ import annotations

import gzip
import io
from collections.abc import Callable
from pathlib import Path

import pandas as pd

#: How much of the file is read to find the header row.
HEAD_BYTES = 1 << 20
#: Tried in this order. latin-1 can read any bytes, so it always works last.
ENCODINGS = ("utf-8", "cp1252", "latin-1")
ENCODING_NAMES = {"cp1252": "Windows-1252", "latin-1": "Latin-1"}


def _head(source: Path | bytes) -> bytes:
    if isinstance(source, bytes):
        return source[:HEAD_BYTES]
    opener = gzip.open if source.name.lower().endswith(".gz") else open
    with opener(source, "rb") as fh:
        return fh.read(HEAD_BYTES)


def guess_separator(header: str, name: str = "") -> str:
    """Tab for .tsv files, else the most common of , ; and tab in the header row."""
    if name.lower().endswith((".tsv", ".tsv.gz", ".tab")):
        return "\t"
    counts = {sep: header.count(sep) for sep in (",", ";", "\t")}
    best = max(counts, key=lambda sep: counts[sep])
    return best if counts[best] else ","


def read_csv(
    source: str | Path | bytes,
    *,
    name: str = "",
    text_only: bool = False,
    max_columns: int | None = None,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Read a CSV file (a path, or the bytes of an upload) into a table.

    ``text_only`` keeps every cell as text, so that it can be checked later.
    ``max_columns`` refuses files with more columns before pandas reads them.
    Raises ValueError (or another pandas error) when the file cannot be read.
    """
    if not isinstance(source, bytes):
        source = Path(source)
        name = name or source.name
    say = log or (lambda _msg: None)
    header = _head(source).split(b"\n", 1)[0].decode("utf-8", errors="replace")
    sep = guess_separator(header, name)
    if max_columns is not None and header.count(sep) + 1 > max_columns:
        raise ValueError(f"the file has more than {max_columns} columns")

    options: dict = {"sep": sep}
    if text_only:
        options.update(dtype=str, keep_default_na=True)
    for encoding in ENCODINGS:
        try:
            data = io.BytesIO(source) if isinstance(source, bytes) else source
            df = pd.read_csv(data, encoding=encoding, **options)
            break
        except UnicodeDecodeError:
            continue
    if encoding != "utf-8":
        say(
            f"Note: {name or 'the file'} is not UTF-8 text, so Harbor read it as {ENCODING_NAMES[encoding]}. "
            'Check the text values, or save the file as "CSV UTF-8 (Comma delimited)".'
        )
    if sep == ";":
        say(f"Note: {name or 'the file'} uses ';' between columns.")
    df.columns = [str(c).strip() for c in df.columns]
    return df
