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
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Import the rawfile parser from the same directory
import sys
sys.path.insert(0, str(Path(__file__).parent))
from parse_rawfile import parse_rawfile, parse_rawfile_header


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

    @property
    def is_ac(self) -> bool:
        return "ac" in self.header.get("plotname", "").lower()

    @property
    def is_transient(self) -> bool:
        pn = self.header.get("plotname", "").lower()
        return "transient" in pn or "tran" in pn

    @property
    def is_dc(self) -> bool:
        pn = self.header.get("plotname", "").lower()
        return "dc" in pn and "ac" not in pn

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
    """
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

    cmd = ["ngspice", "-b", "-r", raw_path, cir_path]
    if extra_flags:
        cmd.extend(extra_flags)

    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )

    # Parse .meas results from stdout
    # ngspice outputs: "name               =  1.59155e+04" for .meas directives
    measurements: dict[str, float] = {}
    # Known ngspice status keys to ignore
    _STATUS_KEYS = {
        "doing analysis at temp", "total analysis time",
        "total elapsed time", "total dram available",
        "dram currently available", "maximum ngspice program size",
        "current ngspice program size", "shared ngspice pages",
        "text (code) pages", "stack", "library pages",
    }
    for line in proc.stdout.splitlines():
        if "=" not in line or line.strip().startswith("*"):
            continue
        parts = line.split("=", 1)
        if len(parts) != 2:
            continue
        name = parts[0].strip().lower()
        # Skip ngspice status/diagnostic lines
        if any(name.startswith(sk) for sk in _STATUS_KEYS):
            continue
        try:
            val = float(parts[1].strip().split()[0])
            measurements[name] = val
        except (ValueError, IndexError):
            pass

    # Parse rawfile
    variables: dict[str, np.ndarray] = {}
    header: dict = {}
    if Path(raw_path).exists():
        variables = parse_rawfile(raw_path)
        header = parse_rawfile_header(raw_path)

    result = SimResult(
        variables=variables,
        header=header,
        netlist_path=cir_path,
        raw_path=raw_path,
        stdout=proc.stdout,
        stderr=proc.stderr,
        returncode=proc.returncode,
        measurements=measurements,
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
        from parse_rawfile import _dump_csv
        # Re-use CSV dumper on the raw file
        import io
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        _dump_csv(result.raw_path)
        sys.stdout = old_stdout
        Path(args.csv).write_text(buf.getvalue())
        print(f"Saved {args.csv}")

    # Cleanup raw file
    Path(result.raw_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
