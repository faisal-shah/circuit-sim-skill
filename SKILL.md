---
name: ngspice
description: >
  Run SPICE circuit simulations with ngspice. Covers netlist authoring,
  AC/DC/transient analysis, binary rawfile parsing, Monte Carlo tolerance
  analysis, temperature sweeps, .meas extraction, and matplotlib plotting.
---

# ngspice Circuit Simulation Skill

Drive ngspice from the command line to simulate analog/mixed-signal circuits.
This skill covers the full workflow: write a netlist, run batch simulation,
parse binary output, and plot results with matplotlib.

## Prerequisites

- `ngspice` installed and on PATH (`ngspice --version` to verify)
- Python 3.10+ with `numpy` and `matplotlib` (use `uv run` with inline metadata)

---

## 1. Netlist Syntax (SPICE3 Format)

```spice
Title Line (REQUIRED — first line is always the title, never a command)
* Comments start with asterisk

* === Component Syntax ===
* Resistor:    Rname node+ node- value
* Capacitor:   Cname node+ node- value
* Inductor:    Lname node+ node- value
* Diode:       Dname anode cathode modelname
* BJT:         Qname collector base emitter modelname
* MOSFET:      Mname drain gate source bulk modelname W=w L=l
* VCVS:        Ename out+ out- ctrl+ ctrl- gain
* Voltage src: Vname node+ node- [DC val] [AC mag [phase]] [transient_func]
* Current src: Iname node+ node- [DC val] [AC mag [phase]]

* === Subcircuit Definition ===
.subckt name port1 port2 ...
* ... components ...
.ends name

* === Parameters ===
.param Rval=1k Cval=100n

* === Models ===
.model NMOS1 NMOS (VTO=0.7 KP=110u)
.model D1N4148 D (IS=2.52e-9 RS=0.568)

* === Include External Files ===
.include "models/opamp.lib"

* === Analysis Commands (pick one or more) ===
.op                              * DC operating point
.dc Vin 0 5 0.1                  * DC sweep: source start stop step
.ac dec 100 1 1e6                * AC sweep: dec/oct/lin Npts fstart fstop
.tran 1u 10m                     * Transient: step stop [start [max_step]]
.step param Rval 50 200 50       * Parameter sweep

* === Measurements ===
.meas tran rise_time TRIG v(out) VAL=0.1 RISE=1 TARG v(out) VAL=0.9 RISE=1
.meas ac f3dB WHEN vdb(out)=-3 FALL=1
.meas dc vout_max MAX v(out)

* === Control Block (batch mode) ===
.control
run
wrdata output.csv v(out) v(in)
write output.raw v(out)
.endc

.end
```

### Key Rules
- **First line is ALWAYS the title** — not a dot-command, not a comment
- **Node `0` is ground** — every circuit must reference node 0
- **Node names are case-insensitive** in standard ngspice
- **Value suffixes:** `f`=1e-15, `p`=1e-12, `n`=1e-9, `u`=1e-6, `m`=1e-3,
  `k`=1e3, `meg`=1e6, `g`=1e9, `t`=1e12
- **SPICE treats `M` as milli** (1e-3), use `MEG` for mega (1e6)

---

## 2. Running ngspice in Batch Mode

Always run batch mode for scripted workflows:

```bash
ngspice -b -r output.raw circuit.cir
```

| Flag | Purpose |
|------|---------|
| `-b` | Batch mode (no interactive prompt) |
| `-r output.raw` | Write binary rawfile (preferred over text) |
| `-o logfile.log` | Redirect stdout/stderr to log |

The `-r` flag writes ALL node voltages and branch currents to the rawfile
automatically — no `.save` or `.write` needed for basic usage.

### Selective Output with .save

To reduce rawfile size, specify which signals to save:

```spice
.save v(out) v(in) i(Vpower)
```

---

## 3. Parsing Binary Rawfiles (Python)

The rawfile is the **primary data exchange format**. Use the helper script
at `scripts/parse_rawfile.py` or inline this pattern:

```python
import struct
import numpy as np
from pathlib import Path

def parse_rawfile(path: str | Path) -> dict[str, np.ndarray]:
    """Parse ngspice binary rawfile → dict of variable_name → complex array.

    For AC analysis: values are complex (magnitude + phase).
    For DC/transient: values are real (imaginary part is zero).
    """
    raw = Path(path).read_bytes()

    # Split header (ASCII) from binary data
    marker = b"Binary:\n"
    hdr_end = raw.index(marker) + len(marker)
    header = raw[:hdr_end].decode(errors="replace")
    data = raw[hdr_end:]

    # Parse header fields
    n_vars = n_pts = None
    varnames: list[str] = []
    is_complex = False
    in_vars = False

    for line in header.splitlines():
        if line.startswith("No. Variables:"):
            n_vars = int(line.split(":", 1)[1])
        elif line.startswith("No. Points:"):
            n_pts = int(line.split(":", 1)[1])
        elif line.startswith("Flags:") and "complex" in line.lower():
            is_complex = True
        elif line.startswith("Variables:"):
            in_vars = True
        elif in_vars:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                varnames.append(parts[1].lower())
            if len(varnames) == n_vars:
                in_vars = False

    # Unpack binary data
    # AC analysis: each value is 2 doubles (real, imag) = 16 bytes
    # DC/tran: each value is 1 double = 8 bytes
    # Exception: the sweep variable (first) is always real in DC/tran,
    # but in AC analysis it's stored as complex too.
    if is_complex:
        # All variables stored as complex (2 × float64)
        values = np.zeros((n_vars, n_pts), dtype=complex)
        offset = 0
        for i in range(n_pts):
            for v in range(n_vars):
                re, im = struct.unpack_from("dd", data, offset)
                values[v, i] = complex(re, im)
                offset += 16
    else:
        # All variables stored as real (1 × float64)
        # Exception: first variable (time/sweep) is float64,
        # remaining are also float64
        values = np.zeros((n_vars, n_pts), dtype=complex)
        offset = 0
        for i in range(n_pts):
            for v in range(n_vars):
                val, = struct.unpack_from("d", data, offset)
                values[v, i] = complex(val, 0)
                offset += 8

    return {name: values[i] for i, name in enumerate(varnames)}
```

### Usage

```python
data = parse_rawfile("output.raw")
freq = np.real(data["frequency"])       # AC sweep variable
vout = data["v(out)"]                   # Complex for AC
mag_dB = 20 * np.log10(np.abs(vout))
phase_deg = np.degrees(np.angle(vout))
```

---

## 4. Analysis Patterns

### 4a. AC Analysis (Bode Plot)

```spice
Bandpass Filter
Vin in 0 AC 1
L1 in mid 1mH
C1 mid out 253nF
R1 out 0 100
.ac dec 100 10 1e6
.end
```

```python
data = parse_rawfile("output.raw")
freq = np.real(data["frequency"])
vout = data["v(out)"]
mag = 20 * np.log10(np.abs(vout) + 1e-30)
phase = np.degrees(np.angle(vout))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.semilogx(freq, mag)
ax1.set_ylabel("Magnitude (dB)")
ax1.axhline(-3, color="red", linestyle="--", label="-3 dB")
ax2.semilogx(freq, phase)
ax2.set_ylabel("Phase (°)")
ax2.set_xlabel("Frequency (Hz)")
```

### 4b. Transient Analysis

```spice
RC Step Response
Vin in 0 PULSE(0 1 0 1n 1n 5m 10m)
R1 in out 1k
C1 out 0 1u
.tran 10u 20m
.end
```

```python
data = parse_rawfile("output.raw")
time = np.real(data["time"])
vout = np.real(data["v(out)"])
plt.plot(time * 1e3, vout)
plt.xlabel("Time (ms)")
```

### 4c. DC Sweep

```spice
Diode IV Curve
Vd anode 0 DC 0
D1 anode 0 DMOD
.model DMOD D (IS=1e-14)
.dc Vd 0 0.8 0.001
.end
```

### 4d. Parameter Sweep with .step

```spice
R Sweep
Vin in 0 AC 1
R1 in out {Rval}
C1 out 0 1n
.param Rval=1k
.step param Rval 500 2k 500
.ac dec 50 1 100MEG
.end
```

The rawfile will contain multiple runs. Parse each run separately:
header will show `No. Points:` for a single run, but the binary section
contains `n_runs × n_pts` points sequentially.

---

## 5. Monte Carlo / Tolerance Analysis

ngspice does not have a built-in Monte Carlo command. Use Python to randomize
component values and run multiple simulations:

```python
import subprocess, tempfile, numpy as np
from pathlib import Path

rng = np.random.default_rng(42)

def make_netlist(r, c, l) -> str:
    return f"""Filter MC Run
Vin in 0 AC 1
L1 in mid {l:.10e}
C1 mid out {c:.10e}
R1 out 0 {r:.6e}
.ac dec 100 100 1e6
.end
"""

# Tolerances: R ±5%, C ±10%, L ±5%
R_NOM, C_NOM, L_NOM = 100, 253e-9, 1e-3

results = []
for i in range(200):
    r = R_NOM * (1 + rng.uniform(-0.05, 0.05))
    c = C_NOM * (1 + rng.uniform(-0.10, 0.10))
    l = L_NOM * (1 + rng.uniform(-0.05, 0.05))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False) as f:
        f.write(make_netlist(r, c, l))
        cir_path = f.name
    raw_path = cir_path.replace(".cir", ".raw")

    subprocess.run(
        ["ngspice", "-b", "-r", raw_path, cir_path],
        capture_output=True, timeout=30,
    )
    data = parse_rawfile(raw_path)
    results.append(data)
    Path(cir_path).unlink()
    Path(raw_path).unlink()
```

### Typical Component Tolerances

| Component | Typical | Precision | Notes |
|-----------|---------|-----------|-------|
| Resistor (metal film) | ±1% | ±0.1% | TC: 25-100 ppm/°C |
| Resistor (carbon) | ±5% | ±1% | TC: 200-500 ppm/°C |
| Capacitor (C0G/NP0) | ±5% | ±1% | TC: ±30 ppm/°C |
| Capacitor (X7R) | ±10% | ±5% | TC: ±15% over range |
| Capacitor (electrolytic) | ±20% | — | Avoid in filters |
| Inductor (ferrite) | ±10% | ±5% | TC: -300 to -800 ppm/°C |

---

## 6. Temperature Sweep

Apply temperature coefficients manually in Python (ngspice's `.temp` only
affects semiconductor models, not passive RLC):

```python
# Temperature coefficients (ppm/°C relative to 25°C)
R_TC  =  100e-6   # metal film resistor
C_TC  = -200e-6   # film capacitor
L_TC  = -400e-6   # ferrite inductor
T_REF = 25.0

def apply_temp(nominal, tc, temp):
    return nominal * (1 + tc * (temp - T_REF))

for temp in [-40, -20, 0, 25, 50, 85, 125, 150]:
    r = apply_temp(R_NOM, R_TC, temp)
    c = apply_temp(C_NOM, C_TC, temp)
    l = apply_temp(L_NOM, L_TC, temp)
    # ... run simulation with adjusted values ...
```

For semiconductor temperature effects, use ngspice's built-in `.temp`:

```spice
.temp 25
* or sweep with:
.step temp -40 150 10
```

---

## 7. Measurements (.meas)

`.meas` extracts scalar metrics from simulation results. They print to
stdout in batch mode — capture and parse:

```spice
.meas ac f_3dB WHEN vdb(out)=-3 FALL=1
.meas ac peak_gain MAX vdb(out)
.meas ac peak_freq AT peak_gain
.meas tran risetime TRIG v(out) VAL=0.1 RISE=1 TARG v(out) VAL=0.9 RISE=1
.meas tran overshoot MAX v(out)
.meas dc vmax MAX v(out)
```

Parse from stdout:

```python
result = subprocess.run(
    ["ngspice", "-b", "circuit.cir"],
    capture_output=True, text=True,
)
for line in result.stdout.splitlines():
    if "f_3db" in line.lower():
        # e.g., "f_3db               =  1.00000e+04"
        val = float(line.split("=")[1])
```

---

## 8. Plotting Conventions

### Standard Bode Plot

```python
import matplotlib
matplotlib.use("Agg")  # headless — always set before importing pyplot
import matplotlib.pyplot as plt

fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax_mag.semilogx(freq, mag_dB, linewidth=2)
ax_mag.axhline(-3, color="red", linestyle="--", linewidth=0.8, label="-3 dB")
ax_mag.set_ylabel("Magnitude (dB)")
ax_mag.grid(True, which="both", alpha=0.3)
ax_mag.legend()

ax_ph.semilogx(freq, phase_deg, linewidth=2, color="tab:orange")
ax_ph.set_ylabel("Phase (°)")
ax_ph.set_xlabel("Frequency (Hz)")
ax_ph.grid(True, which="both", alpha=0.3)

fig.suptitle("Bode Plot", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("bode.png", dpi=150, bbox_inches="tight")
```

### Monte Carlo Overlay

```python
for freq, mag in mc_curves:
    ax.semilogx(freq, mag, color="#1f77b4", alpha=0.05, linewidth=0.5)

# Statistics
all_mags = np.array([m for _, m in mc_curves])
ax.semilogx(freq_ref, np.mean(all_mags, axis=0), "k-", lw=2, label="Mean")
ax.fill_between(freq_ref,
    np.percentile(all_mags, 5, axis=0),
    np.percentile(all_mags, 95, axis=0),
    alpha=0.2, label="5th–95th %ile")
```

### Temperature Sweep with Colormap

```python
cmap = plt.cm.coolwarm
for temp, freq, mag in temp_curves:
    color = cmap((temp - T_MIN) / (T_MAX - T_MIN))
    ax.semilogx(freq, mag, color=color, label=f"{temp}°C")
```

---

## 9. Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| `Error: no circuit loaded` | First line is a dot-command | Add title as first line |
| `Node 0 not found` | No ground reference | Connect something to node `0` |
| `Timestep too small` | Convergence failure in transient | Add `.options reltol=0.003` or use `UIC` |
| `Singular matrix` | Floating node or topology error | Every node needs a DC path to ground |
| `M` means milli not mega | SPICE convention | Use `MEG` for 1e6 |
| Rawfile parse garbage | Text mode vs binary | Always use `-r` flag for binary |
| AC gain > 0 dB for passives | Phase/complex issue | Check `np.abs()` not `.real` |

### Convergence Helpers

```spice
.options reltol=0.003    * Relax tolerance (default 0.001)
.options abstol=1e-10    * Absolute current tolerance
.options vntol=1e-4      * Absolute voltage tolerance
.options itl1=300        * DC iteration limit
.options itl4=50         * Transient iteration limit
.options method=gear     * Integration method (gear or trapezoidal)
```

---

## 10. Script Templates

### PEP 723 Standalone Simulation Script

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Run an ngspice simulation and plot results. Usage: uv run sim.py"""

import subprocess, struct, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

NETLIST = """Title Goes Here
Vin in 0 AC 1
R1 in out 1k
C1 out 0 1n
.ac dec 100 1 100MEG
.end
"""

# Run simulation
with tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False) as f:
    f.write(NETLIST)
    cir = f.name
raw = cir.replace(".cir", ".raw")
subprocess.run(["ngspice", "-b", "-r", raw, cir], capture_output=True)

# Parse rawfile (use scripts/parse_rawfile.py for production code)
# ... (see Section 3) ...

# Plot
# ... (see Section 8) ...

# Cleanup
Path(cir).unlink()
Path(raw).unlink()
```

See `scripts/parse_rawfile.py` and `scripts/run_sim.py` for ready-to-use
helper modules.
