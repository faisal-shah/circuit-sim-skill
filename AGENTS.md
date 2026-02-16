# AGENTS.md — AI Context for ngspice-skill

## What This Repo Is

An agent skill that teaches AI assistants how to drive ngspice for circuit
simulation. The primary artifact is `SKILL.md` which gets loaded into the
AI's context at session start.

## Key Files

- `SKILL.md` — Main skill content. Loaded by the agent framework. Keep it
  concise (under 500 lines) — every line consumes context tokens on every
  invocation.
- `scripts/parse_rawfile.py` — Binary rawfile parser. Library + CLI.
- `scripts/run_sim.py` — End-to-end simulation runner. Handles the .meas
  + rawfile incompatibility automatically.
- `examples/` — Reference netlists for testing.

## Known Gotchas

1. **`.meas` + `-b -r` incompatibility**: ngspice silently suppresses .meas
   output when `-r` is used. `run_sim.py` works around this by injecting a
   `.control` block when .meas directives are detected.

2. **UIC flag**: Without `UIC` on `.tran`, `ic=` values on components are
   silently ignored (ngspice runs DC OP first, which overrides them).

3. **Multi-run rawfiles**: `.step` param sweeps produce concatenated rawfiles.
   Use `parse_rawfile_all()` instead of `parse_rawfile()` for these.

## Testing Changes

```bash
# Run the example netlists through run_sim.py to verify nothing is broken:
uv run scripts/run_sim.py examples/rc_lowpass.cir --plot /tmp/test_ac.png
uv run scripts/run_sim.py examples/pulse_response.cir --plot /tmp/test_tran.png
```

## Style

- SKILL.md: terse, high-signal, no filler. Tables over prose. Code examples
  should be minimal but complete.
- Scripts: PEP 723 inline metadata, type hints, numpy for data.
