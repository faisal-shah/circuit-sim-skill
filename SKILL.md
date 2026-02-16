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
* Capacitor:   Cname node+ node- value [ic=voltage]
* Inductor:    Lname node+ node- value [ic=current]
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
.tran 1u 10m                     * Transient: step stop [start [max_step]] [UIC]
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

**⚠️ `-b -r` suppresses `.meas` results.** ngspice silently discards all `.meas`
output when `-r` is used. If the netlist contains `.meas` directives, either:
1. Use `scripts/run_sim.py` which handles this automatically (injects a `.control` block)
2. Or manually: drop `-r` and use a `.control` block with `run` + `write output.raw`

The `-r` flag writes ALL node voltages and branch currents to the rawfile
automatically — no `.save` or `.write` needed for basic usage.

### Selective Output with .save

To reduce rawfile size, specify which signals to save:

```spice
.save v(out) v(in) i(Vpower)
```

### Initial Conditions and UIC

**This is a critical gotcha.** The `UIC` (Use Initial Conditions) flag on `.tran`
controls whether ngspice uses `ic=` values set on capacitors and inductors.

```spice
* Pre-charged capacitor and inductor with initial current
Cp node+ node- 12u ic=15000
Lp node+ node- 6u ic=0.5
.tran 1u 10m UIC
```

**Without `UIC`** (the default): ngspice computes a DC operating point at t=0
before starting the transient. In the DC operating point, capacitors are open
circuits and inductors are short circuits, so the solver finds the steady-state
DC solution. All `ic=` values on components are **silently ignored**. This
typically produces all-zero waveforms for circuits that depend on stored energy.

**With `UIC`**: ngspice skips the DC operating point entirely and initializes
component voltages/currents directly from `ic=` values at t=0. This is essential
for:
- Pre-charged capacitors (energy storage, pulsed power)
- Oscillator startup from a known state
- Any circuit where the initial stored energy drives the behavior

**`ic=` vs `.ic`** — these are different mechanisms:
- `Cname n+ n- value ic=V` — component-level, **only honored with `UIC`**
- `.ic V(node)=value` — node-level, applied as constraints **after** DC OP
  (does NOT require `UIC`, but behaves differently)

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| All-zero waveforms despite `ic=` on caps | `.tran` missing `UIC` | Add `UIC` to `.tran` line |
| Capacitor starts at 0V not expected voltage | DC OP overrides `ic=` | Add `UIC` to `.tran` line |

---

## 3. Parsing Binary Rawfiles (Python)

The rawfile is the **primary data exchange format**. Use `scripts/parse_rawfile.py`:

```python
from parse_rawfile import parse_rawfile, parse_rawfile_all

data = parse_rawfile("output.raw")
# Returns dict: variable_name → numpy array (complex for AC, real-as-complex for DC/tran)

# For multi-run rawfiles (.step param sweeps, multiple analyses):
runs = parse_rawfile_all("output.raw")  # list of dicts, one per run
```

**Format overview:** ASCII header (title, variable names, flags) → `Binary:\n` marker →
packed little-endian float64s. AC data is stored as complex pairs (16 bytes/value);
DC/transient as real (8 bytes/value). The parser handles both automatically.

### Usage

```python
data = parse_rawfile("output.raw")
freq = np.real(data["frequency"])       # AC sweep variable
vout = data["v(out)"]                   # Complex for AC
mag_dB = 20 * np.log10(np.abs(vout))
phase_deg = np.degrees(np.angle(vout))

# Transient
time = np.real(data["time"])
v_out = np.real(data["v(out)"])         # Real for transient
```

CLI: `uv run scripts/parse_rawfile.py output.raw [--json | --csv]`

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

ngspice has no built-in Monte Carlo. Use Python to randomize and loop:

```python
rng = np.random.default_rng(42)
for i in range(200):
    r = R_NOM * (1 + rng.uniform(-0.05, 0.05))    # ±5%
    c = C_NOM * (1 + rng.uniform(-0.10, 0.10))    # ±10%
    netlist = make_netlist(r, c)  # generate netlist string with adjusted values
    result = simulate(netlist)    # run ngspice, parse rawfile (see scripts/run_sim.py)
    results.append(result)
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

ngspice's `.temp` only affects semiconductor models, not passive RLC. For passives,
apply temperature coefficients manually: `R(T) = R_nom × (1 + TC × (T - 25))`.

| Component | TC (ppm/°C) | Notes |
|-----------|-------------|-------|
| Metal film resistor | +100 | Most stable |
| Film capacitor | -200 | C0G/NP0: ±30 |
| Ferrite inductor | -400 to -800 | Core dependent |

For semiconductors, use ngspice's built-in sweep:

```spice
.temp 25
.step temp -40 150 10    * sweep temperature
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

---

## 9. ngspice Quick Reference

Common elements beyond R/L/C that the AI may need:

| Element | Syntax | Notes |
|---------|--------|-------|
| Coupled inductors | `Kname L1 L2 coupling` | k=0 to 1; define both L elements first |
| V-controlled switch | `Sname n+ n- ctrl+ ctrl- model` | `.model name SW(VT=0 VH=0.1 RON=1 ROFF=1e6)` |
| I-controlled switch | `Wname n+ n- Vctrl model` | `.model name CSW(IT=0 IH=0.1 RON=1 ROFF=1e6)` |
| Behavioral source | `Bname n+ n- V={expr}` | or `I={expr}`; any math on node voltages |
| Ideal transformer | Two `L` + `K` statement | No standalone transformer element |
| VCVS | `Ename out+ out- ctrl+ ctrl- gain` | Ideal voltage amplifier |
| CCCS | `Fname out+ out- Vsense gain` | Current controlled by current through Vsense |
| Diode | `Dname anode cathode model` | `.model name D(IS=1e-14 BV=100)` |
| Transmission line | `Tname p1+ p1- p2+ p2- Z0=50 TD=1n` | Lossless |

**V-controlled switch polarity tip**: the switch closes when `V(ctrl+) - V(ctrl-) > VT`.
For negative output voltages, swap control nodes so the difference is positive.

Full syntax reference: https://ngspice.sourceforge.io/docs/ngspice-manual.pdf

### Transient Source Functions

These are used with `V` or `I` sources. Getting parameter order wrong causes **silent** errors.

```
PULSE(V1 V2 Td Tr Tf PW PER)
  V1=initial, V2=pulsed, Td=delay, Tr=rise, Tf=fall, PW=width, PER=period

SIN(Voff Vamp Freq Td Theta Phase)
  Damped sine: V = Voff + Vamp × sin(2π·Freq·t + Phase) × exp(-Theta·t)

EXP(V1 V2 Td1 Tau1 Td2 Tau2)
  Exponential rise from V1 to V2, then fall

PWL(t1 v1 t2 v2 ...)
  Piecewise linear — arbitrary waveform defined point by point
```

Example: 15 kV pulse with 100 ns rise/fall, 10 µs width, 1 ms period:
```spice
Vpulse in 0 PULSE(0 15000 0 100n 100n 10u 1m)
```

---

## 10. Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| `Error: no circuit loaded` | First line is a dot-command | Add title as first line |
| `Node 0 not found` | No ground reference | Connect something to node `0` |
| `Timestep too small` | Convergence failure in transient | Add `.options reltol=0.003` or use `UIC` |
| `Singular matrix` | Floating node or topology error | Every node needs a DC path to ground |
| `M` means milli not mega | SPICE convention | Use `MEG` for 1e6 |
| Rawfile parse garbage | Text mode vs binary | Always use `-r` flag for binary |
| AC gain > 0 dB for passives | Phase/complex issue | Check `np.abs()` not `.real` |
| `.meas` results missing | `-b -r` suppresses `.meas` | Use `.control` block with `run` + `write` instead of `-r` |
| `ic=` ignored, all zeros | `.tran` without `UIC` | Add `UIC` to `.tran` line |

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

## 11. Helper Scripts

See `scripts/run_sim.py` for a complete PEP 723 simulation runner with CLI,
Bode/transient plots, CSV export, and `.meas` parsing. Usage:

```bash
uv run scripts/run_sim.py circuit.cir --plot bode.png
```

---

## 12. Circuit Visualization

### schemdraw — Quick Inline Schematics

Use `schemdraw` for programmatic circuit diagrams (PNG or SVG output).
PEP 723 deps: `schemdraw`, `matplotlib`, `pillow`.

```python
import schemdraw
import schemdraw.elements as elm

with schemdraw.Drawing() as d:
    d += elm.SourceV().label("Vin").up()
    d += elm.Resistor().right().label("R1")
    d += elm.Capacitor().down().label("C1")
    d += elm.Line().left()
    d.save("circuit.png", dpi=150)
```

**Key gotchas learned from experience:**
- **Transparent background**: schemdraw renders RGBA by default. Composite onto
  white with PIL: `Image.alpha_composite(white_bg, img).save("out.png")`
- **Cursor movement**: `elm.Annotate().at(pos)` moves the drawing cursor. Always
  place `elm.Ground()` with explicit `.at(component.end)` to avoid misconnection.
- **Label overlap on vertical components**: `loc='left'` on vertical inductors/caps
  still overlaps the component body. Use standalone `elm.Annotate()` with explicit
  `(x, y)` coordinates offset 1.3–1.8 units from the component.
- **Title placement**: Render circuit first, then add title via matplotlib
  `fig.suptitle()` to avoid excessive whitespace.

### KiCad Export — Interactive Editing

For schematics that need iterative refinement or production documentation, export
to KiCad format instead of regenerating PNGs each time:

- `kicad-sch-api` package generates `.kicad_sch` files (KiCad 8 compatible)
- Workflow: parse SPICE netlist → map components to KiCad library symbols
  (e.g., `R` → `Device:R`, `C` → `Device:C`) → auto-place on grid → open in KiCad
- No direct SPICE→KiCad converter exists; requires custom mapping code
- `pip install kicad-sch-api` (or `uv pip install`)

This is preferred when the schematic will be revised multiple times or needs to be
included in formal documentation.
