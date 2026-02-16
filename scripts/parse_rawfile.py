# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Parse ngspice binary rawfiles into numpy arrays.

Usage:
    uv run parse_rawfile.py output.raw              # print summary
    uv run parse_rawfile.py output.raw --json       # dump as JSON
    uv run parse_rawfile.py output.raw --csv        # dump as CSV

As a library:
    from parse_rawfile import parse_rawfile
    data = parse_rawfile("output.raw")
    freq = np.real(data["frequency"])
    vout = data["v(out)"]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def parse_rawfile(path: str | Path) -> dict[str, np.ndarray]:
    """Parse an ngspice binary rawfile (first plot/run).

    Returns a dict mapping lowercase variable names to numpy arrays.
    AC analysis produces complex arrays; DC/transient produce real-valued
    arrays (stored as complex with zero imaginary part for uniformity).

    For rawfiles with multiple runs (e.g. from .step), use parse_rawfile_all().
    """
    raw = Path(path).read_bytes()
    return _parse_single_plot(raw, 0)[0]


def parse_rawfile_all(path: str | Path) -> list[dict[str, np.ndarray]]:
    """Parse all runs/plots from a multi-run rawfile (e.g. .step param sweeps).

    Returns a list of dicts, one per run. Single-run rawfiles return a 1-element list.
    """
    raw = Path(path).read_bytes()
    results = []
    offset = 0
    while offset < len(raw):
        data, next_offset = _parse_single_plot(raw, offset)
        results.append(data)
        if next_offset <= offset:
            break
        offset = next_offset
    return results


def _parse_single_plot(raw: bytes, start: int) -> tuple[dict[str, np.ndarray], int]:
    """Parse one plot from raw bytes starting at `start`. Returns (data, next_offset)."""
    marker = b"Binary:\n"
    idx = raw.index(marker, start)
    header = raw[start : idx + len(marker)].decode(errors="replace")
    bin_start = idx + len(marker)

    n_vars: int | None = None
    n_pts: int | None = None
    is_complex = False
    varnames: list[str] = []
    in_vars = False

    for line in header.splitlines():
        low = line.strip().lower()
        if line.startswith("No. Variables:"):
            n_vars = int(line.split(":", 1)[1])
        elif line.startswith("No. Points:"):
            n_pts = int(line.split(":", 1)[1])
        elif line.startswith("Flags:"):
            is_complex = "complex" in low
        elif line.startswith("Variables:"):
            in_vars = True
        elif in_vars:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                varnames.append(parts[1].lower())
            if n_vars is not None and len(varnames) == n_vars:
                in_vars = False

    assert n_vars is not None and n_pts is not None, "Malformed rawfile header"
    assert len(varnames) == n_vars, f"Expected {n_vars} vars, found {len(varnames)}"

    values = np.zeros((n_vars, n_pts), dtype=complex)
    offset = bin_start

    if is_complex:
        for i in range(n_pts):
            for v in range(n_vars):
                re, im = struct.unpack_from("dd", raw, offset)
                values[v, i] = complex(re, im)
                offset += 16
    else:
        for i in range(n_pts):
            for v in range(n_vars):
                (val,) = struct.unpack_from("d", raw, offset)
                values[v, i] = complex(val, 0)
                offset += 8

    return {name: values[i] for i, name in enumerate(varnames)}, offset


def parse_rawfile_header(path: str | Path) -> dict:
    """Parse only the header of a rawfile (no data). Useful for inspection."""
    raw = Path(path).read_bytes()
    marker = b"Binary:\n"
    header = raw[: raw.index(marker)].decode(errors="replace")

    info: dict = {"variables": [], "flags": ""}
    in_vars = False
    for line in header.splitlines():
        if line.startswith("Title:"):
            info["title"] = line.split(":", 1)[1].strip()
        elif line.startswith("Plotname:"):
            info["plotname"] = line.split(":", 1)[1].strip()
        elif line.startswith("Flags:"):
            info["flags"] = line.split(":", 1)[1].strip()
        elif line.startswith("No. Variables:"):
            info["n_vars"] = int(line.split(":", 1)[1])
        elif line.startswith("No. Points:"):
            info["n_pts"] = int(line.split(":", 1)[1])
        elif line.startswith("Variables:"):
            in_vars = True
        elif in_vars:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0].isdigit():
                info["variables"].append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "type": parts[2],
                })
    return info


# ── CLI ──────────────────────────────────────────────────────────────────

def _print_summary(path: str) -> None:
    info = parse_rawfile_header(path)
    print(f"Title:      {info.get('title', '?')}")
    print(f"Plot:       {info.get('plotname', '?')}")
    print(f"Flags:      {info.get('flags', '')}")
    print(f"Variables:  {info.get('n_vars', '?')}")
    print(f"Points:     {info.get('n_pts', '?')}")
    print()
    for v in info.get("variables", []):
        print(f"  {v['index']:3d}  {v['name']:<20s}  {v['type']}")


def _dump_json(path: str) -> None:
    data = parse_rawfile(path)
    out = {}
    for name, arr in data.items():
        if np.all(arr.imag == 0):
            out[name] = arr.real.tolist()
        else:
            out[name] = {"real": arr.real.tolist(), "imag": arr.imag.tolist()}
    json.dump(out, sys.stdout, indent=2)
    print()


def dump_csv(path: str) -> str:
    """Return CSV content as a string from a rawfile."""
    data = parse_rawfile(path)
    names = list(data.keys())
    is_complex = any(np.any(data[n].imag != 0) for n in names)
    lines: list[str] = []

    if is_complex:
        header_parts = []
        for n in names:
            header_parts.extend([f"{n}_re", f"{n}_im"])
        lines.append(",".join(header_parts))
        n_pts = len(data[names[0]])
        for i in range(n_pts):
            row = []
            for n in names:
                row.extend([f"{data[n][i].real:.10e}", f"{data[n][i].imag:.10e}"])
            lines.append(",".join(row))
    else:
        lines.append(",".join(names))
        n_pts = len(data[names[0]])
        for i in range(n_pts):
            lines.append(",".join(f"{data[n][i].real:.10e}" for n in names))

    return "\n".join(lines) + "\n"


# Keep old name as alias for backward compatibility
_dump_csv = dump_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse ngspice binary rawfile")
    parser.add_argument("rawfile", help="Path to .raw file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    args = parser.parse_args()

    if args.json:
        _dump_json(args.rawfile)
    elif args.csv:
        print(dump_csv(args.rawfile), end="")
    else:
        _print_summary(args.rawfile)


if __name__ == "__main__":
    main()
