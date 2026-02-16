# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
End-to-end ngspice simulation runner.

Accepts a netlist (file or string), runs ngspice in batch mode, parses the
binary rawfile, and returns results as numpy arrays. Optionally generates
a Bode plot or time-domain plot.

Usage:
    uv run run_sim.py circuit.cir                     # run + print summary
    uv run run_sim.py circuit.cir --plot bode.png     # run + save Bode plot
    uv run run_sim.py circuit.cir --csv results.csv   # run + export CSV

As a library:
    from run_sim import simulate
    result = simulate("circuit.cir")
    print(result.variables)  # dict of name → numpy array
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Import the rawfile parser from the same directory
import sys
sys.path.insert(0, str(Path(__file__).parent))
from parse_rawfile import parse_rawfile, parse_rawfile_all, parse_rawfile_header, dump_csv


@dataclass
class SimResult:
    """Container for simulation results."""
    variables: dict[str, np.ndarray]
    header: dict
    netlist_path: str
    raw_path: str
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    measurements: dict[str, float] = field(default_factory=dict)
    all_runs: list[dict[str, np.ndarray]] = field(default_factory=list)

    @property
    def is_ac(self) -> bool:
        return "ac" in self.header.get("plotname", "").lower()

    @property
    def is_transient(self) -> bool:
        pn = self.header.get("plotname", "").lower()
        return "transient" in pn or "tran" in pn

    @property
    def is_op(self) -> bool:
        pn = self.header.get("plotname", "").lower()
        return "operating point" in pn

    @property
    def sweep_var(self) -> np.ndarray:
        """Return the independent variable (frequency, time, or voltage)."""
        first_name = list(self.variables.keys())[0]
        return np.real(self.variables[first_name])

    def mag_dB(self, node: str) -> np.ndarray:
        """Magnitude in dB for a given node (AC analysis)."""
        return 20 * np.log10(np.abs(self.variables[node]) + 1e-30)

    def phase_deg(self, node: str) -> np.ndarray:
        """Phase in degrees for a given node (AC analysis)."""
        return np.degrees(np.angle(self.variables[node]))

    def real(self, node: str) -> np.ndarray:
        """Real-valued signal (transient/DC)."""
        return np.real(self.variables[node])


def _netlist_has_meas(netlist_text: str) -> bool:
    """Check if a netlist contains .meas/.measure directives."""
    return bool(re.search(r'^\s*\.meas(ure)?\s', netlist_text, re.MULTILINE | re.IGNORECASE))


def _netlist_has_step(netlist_text: str) -> bool:
    """Check if a netlist contains a .step directive."""
    return bool(re.search(r'^\s*\.step\s', netlist_text, re.MULTILINE | re.IGNORECASE))


_SPICE_SUFFIXES = {
    "t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3,
    "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
}


def _spice_float(s: str) -> float:
    """Parse a SPICE numeric literal (e.g. '1k', '10n', '2.5MEG')."""
    s = s.strip().lower()
    for suf, mult in sorted(_SPICE_SUFFIXES.items(), key=lambda x: -len(x[0])):
        if s.endswith(suf):
            return float(s[: -len(suf)]) * mult
    return float(s)


def _parse_step_directive(netlist_text: str) -> tuple[str, list[float]] | None:
    """Parse '.step param <name> <start> <stop> <incr>' and return (name, values)."""
    m = re.search(
        r'^\s*\.step\s+param\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)',
        netlist_text, re.MULTILINE | re.IGNORECASE,
    )
    if not m:
        return None
    name = m.group(1)
    start, stop, incr = _spice_float(m.group(2)), _spice_float(m.group(3)), _spice_float(m.group(4))
    vals: list[float] = []
    v = start
    while v <= stop * (1 + 1e-9):
        vals.append(v)
        v += incr
    return name, vals


def _inject_step_control_block(netlist_text: str, raw_path: str) -> str:
    """Replace .step directive with a .control foreach loop that writes a multi-run rawfile."""
    parsed = _parse_step_directive(netlist_text)
    if parsed is None:
        return netlist_text
    param_name, values = parsed
    raw_esc = raw_path.replace("\\", "/")
    val_str = " ".join(f"{v:g}" for v in values)
    control = (
        f".control\nset appendwrite\n"
        f"foreach __val {val_str}\n"
        f"  alterparam {param_name} = $__val\n"
        f"  reset\n  run\n  write {raw_esc}\nend\nquit\n.endc\n"
    )
    # Remove the .step line
    text = re.sub(r'^\s*\.step\s.*$', '', netlist_text, count=1, flags=re.MULTILINE | re.IGNORECASE)
    # Insert control block before .end
    end_match = list(re.finditer(r'^\s*\.end\s*$', text, re.MULTILINE | re.IGNORECASE))
    if not end_match:
        return text + "\n" + control + ".end\n"
    last_end = end_match[-1]
    return text[:last_end.start()] + control + text[last_end.start():]


def _netlist_has_control(netlist_text: str) -> bool:
    """Check if a netlist already contains a .control block."""
    return bool(re.search(r'^\s*\.control\b', netlist_text, re.MULTILINE | re.IGNORECASE))


def _check_uic_warning(netlist_text: str) -> str | None:
    """Warn if ic= values exist on components but .tran is missing UIC."""
    has_ic = bool(re.search(r'^[a-zA-Z]\S*\s+.*\bic=', netlist_text, re.MULTILINE | re.IGNORECASE))
    if not has_ic:
        return None
    tran_match = re.search(r'^\s*\.tran\b(.*)$', netlist_text, re.MULTILINE | re.IGNORECASE)
    if tran_match and not re.search(r'\bUIC\b', tran_match.group(1), re.IGNORECASE):
        return (
            "Netlist has ic= values on components but .tran is missing UIC flag.\n"
            "Without UIC, ngspice computes a DC operating point first, silently ignoring all ic= values.\n"
            "Add 'UIC' to the .tran line if initial conditions should be used."
        )
    return None


def _inject_control_block(netlist_text: str, raw_path: str) -> str:
    """Inject a .control block before .end to run simulation and write rawfile.

    This is needed when the netlist has .meas directives, because ngspice's
    -b -r mode silently suppresses all .meas output. The .control block
    approach runs interactively (within batch) and writes the rawfile via
    the `write` command, allowing .meas to work.
    """
    raw_esc = raw_path.replace("\\", "/")
    control = (
        f".control\nrun\nwrite {raw_esc}\nquit\n.endc\n"
    )
    # Insert before the last .end
    end_match = list(re.finditer(r'^\s*\.end\s*$', netlist_text, re.MULTILINE | re.IGNORECASE))
    if not end_match:
        return netlist_text + "\n" + control + ".end\n"
    last_end = end_match[-1]
    return netlist_text[:last_end.start()] + control + netlist_text[last_end.start():]


_STATUS_KEYS = {
    "doing analysis at temp", "total analysis time",
    "total elapsed time", "total dram available",
    "dram currently available", "maximum ngspice program size",
    "current ngspice program size", "shared ngspice pages",
    "text (code) pages", "stack", "library pages",
}


def _parse_measurements(stdout: str) -> dict[str, float]:
    """Parse .meas results from ngspice stdout.

    ngspice outputs lines like: "name  =  1.59155e+04 targ= ... trig= ..."
    """
    measurements: dict[str, float] = {}
    for line in stdout.splitlines():
        if "=" not in line or line.strip().startswith("*"):
            continue
        parts = line.split("=", 1)
        if len(parts) != 2:
            continue
        name = parts[0].strip().lower()
        if any(name.startswith(sk) for sk in _STATUS_KEYS):
            continue
        try:
            val = float(parts[1].strip().split()[0])
            measurements[name] = val
        except (ValueError, IndexError):
            pass
    return measurements


def simulate(
    netlist: str | Path,
    *,
    timeout: int = 60,
    extra_flags: list[str] | None = None,
) -> SimResult:
    """Run an ngspice simulation and return parsed results.

    Args:
        netlist: Path to a .cir file, or a netlist string.
        timeout: Max seconds to wait for ngspice.
        extra_flags: Additional ngspice command-line flags.

    Returns:
        SimResult with parsed data, stdout, stderr, measurements.

    Raises:
        FileNotFoundError: If ngspice is not installed / not on PATH.
    """
    if shutil.which("ngspice") is None:
        raise FileNotFoundError(
            "ngspice not found on PATH. Install it from "
            "https://ngspice.sourceforge.io/ and ensure it's in your PATH."
        )

    # Handle string netlist
    cleanup_cir = False
    if isinstance(netlist, str) and not Path(netlist).exists():
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".cir", delete=False
        )
        tmp.write(netlist)
        tmp.close()
        cir_path = tmp.name
        cleanup_cir = True
    else:
        cir_path = str(netlist)

    raw_path = cir_path.rsplit(".", 1)[0] + ".raw"
    netlist_text = Path(cir_path).read_text()

    uic_warning = _check_uic_warning(netlist_text)
    if uic_warning:
        print(f"WARNING: {uic_warning}", file=sys.stderr)

    has_meas = _netlist_has_meas(netlist_text)
    has_step = _netlist_has_step(netlist_text)

    if (has_meas or has_step) and not _netlist_has_control(netlist_text):
        if has_step:
            injected = _inject_step_control_block(netlist_text, raw_path)
        else:
            injected = _inject_control_block(netlist_text, raw_path)
        inj_tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".cir", delete=False
        )
        inj_tmp.write(injected)
        inj_tmp.close()
        cmd = ["ngspice", "-b", inj_tmp.name]
        cleanup_inj = True
    else:
        cmd = ["ngspice", "-b", "-r", raw_path, cir_path]
        cleanup_inj = False

    if extra_flags:
        cmd.extend(extra_flags)

    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )

    if cleanup_inj:
        Path(inj_tmp.name).unlink(missing_ok=True)

    # Parse .meas results from stdout
    measurements = _parse_measurements(proc.stdout)

    # Parse rawfile
    variables: dict[str, np.ndarray] = {}
    all_runs: list[dict[str, np.ndarray]] = []
    header: dict = {}
    if Path(raw_path).exists():
        header = parse_rawfile_header(raw_path)
        if _netlist_has_step(netlist_text):
            all_runs = parse_rawfile_all(raw_path)
            variables = all_runs[0] if all_runs else {}
        else:
            variables = parse_rawfile(raw_path)
            all_runs = [variables]

    result = SimResult(
        variables=variables,
        header=header,
        netlist_path=cir_path,
        raw_path=raw_path,
        stdout=proc.stdout,
        stderr=proc.stderr,
        returncode=proc.returncode,
        measurements=measurements,
        all_runs=all_runs,
    )

    if cleanup_cir:
        Path(cir_path).unlink(missing_ok=True)

    return result


def plot_bode(result: SimResult, output: str, nodes: list[str] | None = None) -> None:
    """Generate a Bode plot from AC analysis results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freq = result.sweep_var

    if nodes is None:
        # Auto-detect: output voltage nodes (exclude sweep var and v(in))
        first = list(result.variables.keys())[0]
        nodes = [
            n for n in result.variables
            if n != first and n.startswith("v(") and n != "v(in)"
        ]
        if not nodes:
            # Fallback: all voltage nodes except sweep
            nodes = [n for n in result.variables if n != first and n.startswith("v(")]

    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        result.header.get("title", "Bode Plot"), fontsize=14, fontweight="bold"
    )

    if len(result.all_runs) > 1:
        for run_idx, run_vars in enumerate(result.all_runs):
            run_freq = np.real(run_vars[list(run_vars.keys())[0]])
            for node in nodes:
                mag = 20 * np.log10(np.abs(run_vars[node]) + 1e-30)
                phase = np.degrees(np.angle(run_vars[node]))
                lbl = f"{node} run {run_idx + 1}"
                ax_mag.semilogx(run_freq, mag, linewidth=2, alpha=0.6, label=lbl)
                ax_ph.semilogx(run_freq, phase, linewidth=2, alpha=0.6, label=lbl)
    else:
        for node in nodes:
            mag = result.mag_dB(node)
            phase = result.phase_deg(node)
            ax_mag.semilogx(freq, mag, linewidth=2, label=node)
            ax_ph.semilogx(freq, phase, linewidth=2, label=node)

    ax_mag.axhline(-3, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(loc="best", fontsize=9)

    ax_ph.set_ylabel("Phase (°)")
    ax_ph.set_xlabel("Frequency (Hz)")
    ax_ph.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


def plot_transient(
    result: SimResult, output: str, nodes: list[str] | None = None
) -> None:
    """Generate a time-domain plot from transient analysis results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = result.sweep_var

    if nodes is None:
        first = list(result.variables.keys())[0]
        nodes = [
            n for n in result.variables
            if n != first and n.startswith("v(") and n != "v(in)"
        ]
        if not nodes:
            nodes = [n for n in result.variables if n != first and n.startswith("v(")]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        result.header.get("title", "Transient Analysis"),
        fontsize=14,
        fontweight="bold",
    )

    # Auto-scale time axis
    t_max = time[-1]
    if t_max < 1e-6:
        t_scale, t_unit = 1e9, "ns"
    elif t_max < 1e-3:
        t_scale, t_unit = 1e6, "µs"
    elif t_max < 1:
        t_scale, t_unit = 1e3, "ms"
    else:
        t_scale, t_unit = 1, "s"

    if len(result.all_runs) > 1:
        for run_idx, run_vars in enumerate(result.all_runs):
            run_time = np.real(run_vars[list(run_vars.keys())[0]])
            for node in nodes:
                lbl = f"{node} run {run_idx + 1}"
                ax.plot(run_time * t_scale, np.real(run_vars[node]),
                        linewidth=2, alpha=0.6, label=lbl)
    else:
        for node in nodes:
            ax.plot(time * t_scale, result.real(node), linewidth=2, label=node)

    ax.set_xlabel(f"Time ({t_unit})")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run ngspice simulation")
    parser.add_argument("netlist", help="Path to .cir netlist file")
    parser.add_argument("--plot", metavar="FILE", help="Save plot to FILE")
    parser.add_argument("--csv", metavar="FILE", help="Export results to CSV")
    parser.add_argument(
        "--nodes", nargs="+", help="Nodes to plot/export (default: all v(*))"
    )
    args = parser.parse_args()

    result = simulate(args.netlist)

    if result.returncode != 0:
        print(f"ngspice failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # Print summary
    print(f"Analysis: {result.header.get('plotname', '?')}")
    print(f"Points:   {result.header.get('n_pts', '?')}")
    print(f"Variables: {', '.join(result.variables.keys())}")
    if len(result.all_runs) > 1:
        print(f"Runs:     {len(result.all_runs)}")

    # .op: print operating point table
    if result.is_op:
        print("\nOperating Point:")
        for name, arr in result.variables.items():
            val = np.real(arr[0])
            if abs(val) < 1e-3 and val != 0:
                print(f"  {name:<20s} = {val:.6e}")
            else:
                print(f"  {name:<20s} = {val:.6f}")

    if result.measurements:
        print("\nMeasurements:")
        for name, val in result.measurements.items():
            print(f"  {name} = {val:.6e}")

    # Plot
    if args.plot:
        if result.is_ac:
            plot_bode(result, args.plot, args.nodes)
        elif result.is_transient:
            plot_transient(result, args.plot, args.nodes)
        else:
            print(f"Auto-plot not supported for {result.header.get('plotname')}")

    # CSV export
    if args.csv:
        Path(args.csv).write_text(dump_csv(result.raw_path))
        print(f"Saved {args.csv}")

    # Cleanup raw file
    Path(result.raw_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
