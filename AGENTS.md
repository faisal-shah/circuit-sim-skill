# AGENTS.md — AI Context for circuit-sim-skill

## What This Repo Is

An agent skill that teaches AI assistants how to drive ngspice for circuit
simulation. The primary artifact is `SKILL.md` which gets loaded into the
AI's context at session start.

## Key Files

- `SKILL.md` — Main skill content. Loaded by the agent framework. Keep it
  concise (under 500 lines) — every line consumes context tokens on every
  invocation.
- `scripts/parse_rawfile.py` — Binary rawfile parser. Library + CLI.
- `scripts/run_sim.py` — End-to-end simulation runner. Handles .meas, .step
  param sweeps, and UIC warnings automatically.
- `scripts/compile_tex.py` — Compiles Circuitikz `.tex` schematics to PNG
  via pdflatex + pdftoppm.
- `examples/` — Reference netlists for testing.

## Known Gotchas (all handled by tooling)

1. **`.meas` + `-b -r` incompatibility**: ngspice silently suppresses .meas
   output when `-r` is used. **Handled:** `run_sim.py` auto-injects a
   `.control` block when .meas directives are detected.

2. **UIC flag**: Without `UIC` on `.tran`, `ic=` values on components are
   silently ignored (ngspice runs DC OP first, which overrides them).
   **Handled:** `run_sim.py` prints a warning to stderr when `ic=` is found
   without `UIC`. Not auto-fixed because skipping UIC can be intentional.

3. **Multi-run rawfiles**: `.step` param sweeps produce concatenated rawfiles.
   **Handled:** `run_sim.py` auto-detects `.step` directives, injects a
   `.control` foreach loop, and populates `result.all_runs`. Plotting
   functions overlay all runs automatically.

## Schematic Approach

**Use Circuitikz (LaTeX)** for circuit schematics — hand-written coordinate-based
layout produces publication-quality output. schemdraw and auto-layout tools
(SKiDL+netlistsvg) were evaluated and found inadequate for non-trivial circuits.

`scripts/compile_tex.py` handles the pdflatex → PNG pipeline.

## Testing Changes

```bash
# Run the example netlists through run_sim.py to verify nothing is broken:
uv run scripts/run_sim.py examples/rc_lowpass.cir --plot /tmp/test_ac.png
uv run scripts/run_sim.py examples/pulse_response.cir --plot /tmp/test_tran.png
uv run scripts/run_sim.py examples/rc_sweep.cir --plot /tmp/test_sweep.png
```

## Style

- SKILL.md: terse, high-signal, no filler. Tables over prose. Code examples
  should be minimal but complete.
- Scripts: PEP 723 inline metadata, type hints, numpy for data.
