---
name: circuit-sim
description: >
  Run SPICE circuit simulations with ngspice. Covers netlist authoring,
  AC/DC/transient analysis, binary rawfile parsing, Monte Carlo tolerance
  analysis, temperature sweeps, .meas extraction, and matplotlib plotting.
  For converting netlists to schematic diagrams, see the netlist-to-schematic
  skill.
---

# Circuit Simulation Skill

Drive ngspice from the command line to simulate analog/mixed-signal circuits.
This skill covers the full workflow: write a netlist, run batch simulation,
parse binary output, and plot results with matplotlib.

> **Schematic diagrams:** To convert a netlist into a circuit schematic, use
> the `netlist-to-schematic` skill instead.

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

**Preferred: `.control` block** — embed run + write commands directly in the netlist.
This is the most reliable pattern and works with `.meas` directives:

```spice
.control
run
write output.raw
.endc
```

Run with: `ngspice -b circuit.cir`

**Simple alternative:** `ngspice -b -r output.raw circuit.cir` writes all signals
automatically — but **silently suppresses `.meas` results**. Use only without `.meas`.

`scripts/run_sim.py` handles this automatically — detects `.meas` and injects
a `.control` block when needed.

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

**Without `UIC`** (default): ngspice computes a DC operating point first —
capacitors open, inductors shorted — and **silently ignores** all `ic=` values.
This produces all-zero waveforms for stored-energy circuits.

**With `UIC`**: skips DC OP, initializes directly from `ic=` values. Essential for
pre-charged capacitors, oscillator startup, or any energy-storage circuit.

**`ic=` vs `.ic`**: `ic=` on components requires `UIC`; `.ic V(node)=val` is a
post-DC-OP constraint (different mechanism, does NOT require `UIC`).

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

AC data is complex; DC/transient is real (stored as complex with zero imaginary).
Use `np.real()` for time/DC values, `np.abs()`/`np.angle()` for AC.

```python
data = parse_rawfile("output.raw")
time = np.real(data["time"])        # Transient
vout_ac = data["v(out)"]           # AC: complex → use np.abs(), np.angle()
mag_dB = 20 * np.log10(np.abs(vout_ac))
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

### 4b. Transient Analysis

```spice
RC Step Response
Vin in 0 PULSE(0 1 0 1n 1n 5m 10m)
R1 in out 1k
C1 out 0 1u
.tran 10u 20m
.end
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

The rawfile contains multiple runs. Use `parse_rawfile_all()` to get a list of
dicts, one per run. `run_sim.py` handles this automatically with `result.all_runs`.

---

## 5. Monte Carlo / Tolerance Analysis

ngspice has no built-in Monte Carlo. Use Python to randomize and loop:

```python
rng = np.random.default_rng(42)
for i in range(200):
    r = R_NOM * (1 + rng.uniform(-0.05, 0.05))    # ±5%
    c = C_NOM * (1 + rng.uniform(-0.10, 0.10))    # ±10%
    netlist = make_netlist(r, c)
    results.append(simulate(netlist))
```

### Component Tolerances & Temperature Coefficients

| Component | Tolerance | TC (ppm/°C) |
|-----------|-----------|-------------|
| Resistor (metal film) | ±1% | +25 to +100 |
| Resistor (carbon) | ±5% | +200 to +500 |
| Capacitor (C0G/NP0) | ±5% | ±30 |
| Capacitor (X7R) | ±10% | ±15% over range |
| Capacitor (electrolytic) | ±20% | — |
| Inductor (ferrite) | ±10% | -400 to -800 |

---

## 6. Temperature Sweep

ngspice `.temp`/`.step temp` only affects semiconductor models, not passive RLC.
For passives, apply TC manually: `R(T) = R_nom × (1 + TC × (T - 25))`.

```spice
.step temp -40 150 10    * sweep semiconductor temperature
```

---

## 7. Measurements (.meas)

`.meas` extracts scalar metrics. `run_sim.py` parses them automatically.

```spice
.meas ac f_3dB WHEN vdb(out)=-3 FALL=1
.meas tran risetime TRIG v(out) VAL=0.1 RISE=1 TARG v(out) VAL=0.9 RISE=1
.meas tran overshoot MAX v(out)
```

Manual parsing from stdout: look for `meas_name = value` lines.

---

## 8. Plotting Conventions

Always set the Agg backend **before** importing pyplot — headless environments
hang or error otherwise:

```python
import matplotlib
matplotlib.use("Agg")  # must come before pyplot import
import matplotlib.pyplot as plt
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

Full syntax reference: https://ngspice.sourceforge.io/docs/ngspice-html-manual/manual.xhtml

### Coupled Inductors (Transformers)

ngspice has no standalone transformer element. Model with two inductors + a K
coupling statement. Turns ratio ≈ √(L2/L1).

```spice
* Iron-core power transformer: 1:10 turns ratio, k=0.95
Lpri  pri_top  pri_bot  1m
Lsec  sec_top  sec_bot  100m
KTR   Lpri Lsec 0.95
```

Both inductors support `ic=` for initial current (requires `UIC` on `.tran`).
Define both `L` elements **before** the `K` statement.

### V-Controlled Switch Patterns

```spice
* Basic switch: closes when V(ctrl) - V(0) > VT
Smain  out  load  ctrl  0  SWMOD
.model SWMOD SW(VT=0.5 VH=0.1 RON=5m ROFF=1MEG)

* Threshold switch: closes when voltage exceeds threshold
* For negative voltages, swap ctrl+/ctrl- so difference is positive:
Sbrk  node  gnd  0  node  BRKMOD
.model BRKMOD SW(VT=100 VH=5 RON=1 ROFF=1G)
```

### Netlist Structure for Complex Circuits

Use `.param` and section comments for readability:

```spice
Title — Circuit Name
.param Vsrc=12 Lp=1m Ls=100m Cload=10u

* === Source ===
Vsrc  src  0  DC {Vsrc}
Lpri  src  xfmr_p  {Lp}

* === Load ===
Lsec  xfmr_s  out  {Ls}
KTR   Lpri Lsec 0.95
Cload out  0  {Cload}

.control
run
write output.raw
.endc
.end
```

### Subcircuit Usage

Define reusable blocks with `.subckt` and instantiate with `X`:

```spice
* Define a voltage regulator subcircuit
.subckt LDO in out gnd
R1   in  mid  10
C1   mid gnd  1u
Breg out gnd  V={min(V(mid,gnd), 3.3)}
.ends LDO

* Instantiate it
X1  vin  vout  0  LDO
X2  vin  vout2 0  LDO
```

Pin order in `Xname` must match `.subckt` port order exactly. Internal node
names are local to each instance (no collisions between X1 and X2).

### Behavioral Sources (B Element)

Model nonlinear or computed quantities with arbitrary expressions:

```spice
* Voltage limiter (clamp to ±5V)
Blim out 0 V={max(-5, min(5, V(in)))}

* Absolute value rectifier
Babs out 0 V={abs(V(in))}

* Power computation (V × I)
Bpwr pwr 0 V={V(load)*I(Vsense)}
```

Expressions can reference any node voltage `V(node)` or branch current
`I(Vsource)`. Supports standard math functions: `abs`, `sqrt`, `exp`,
`log`, `sin`, `cos`, `min`, `max`, `atan2`, `pow`, ternary `(cond ? a : b)`.

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

Example: 5V pulse with 10ns rise/fall, 10µs width, 100µs period:
```spice
Vpulse in 0 PULSE(0 5 0 10n 10n 10u 100u)
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
| `.meas` results missing | `-b -r` suppresses `.meas` | Use `run_sim.py` (auto-handled) or `.control` block with `run` + `write` |
| `ic=` ignored, all zeros | `.tran` without `UIC` | Add `UIC` to `.tran` line (`run_sim.py` warns automatically) |

### Convergence Helpers

Escalation order: `method=gear` → `reltol=0.003` → `itl4=50` → relax `abstol`/`vntol`.

| Option | Default | When to change |
|--------|---------|----------------|
| `method=gear` | trapezoidal | Stiff circuits: switching, high-Q, large L/C ratios |
| `reltol` | 0.001 | Relax to 0.003 if "timestep too small" |
| `itl1` / `itl4` | 100 / 10 | Increase to 300/50 if convergence iterations fail |
| `abstol` / `vntol` | 1e-12 / 1e-6 | Relax for large-signal circuits (kV/kA range) |

```spice
.options reltol=0.003 method=gear itl4=50
```

---

## 11. Helper Scripts

- `scripts/run_sim.py` — Full simulation runner with auto-handling of `.meas`,
  `.step` param sweeps, and UIC warnings. Bode/transient plots, CSV export.
- `scripts/parse_rawfile.py` — Binary rawfile parser (single + multi-run).

Usage:

```bash
uv run scripts/run_sim.py circuit.cir --plot bode.png
uv run scripts/parse_rawfile.py output.raw [--json | --csv]
```


