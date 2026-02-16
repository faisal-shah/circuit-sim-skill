# ngspice-skill

An [agent skill](https://docs.github.com/copilot/concepts/agents/about-agent-skills)
that teaches AI coding assistants how to drive **ngspice** for analog circuit
simulation — from netlist authoring through binary rawfile parsing to
publication-quality plots.

## What's Included

| File | Purpose |
|------|---------|
| `SKILL.md` | Main skill file — netlist syntax, analysis patterns, rawfile parsing, Monte Carlo, plotting, visualization |
| `scripts/parse_rawfile.py` | Standalone binary rawfile parser (CLI + library) |
| `scripts/run_sim.py` | End-to-end sim runner: netlist → rawfile → numpy arrays → plots |

## Installation

### GitHub Copilot / VS Code

Clone and copy the skill:

```bash
git clone https://github.com/faisal-shah/ngspice-skill.git
```

Then install into your project or user-level skills:

```bash
# Option 1: project-level
mkdir -p .github/skills
cp -r ngspice-skill .github/skills/ngspice

# Option 2: user-level (all projects)
cp -r ngspice-skill ~/.copilot/skills/ngspice
```

### Claude Code

```bash
cp -r /path/to/ngspice-skill ~/.claude/skills/ngspice
```

### OpenAI Codex

```bash
cp -r /path/to/ngspice-skill ~/.codex/skills/ngspice
```

## Prerequisites

- **ngspice** installed and on PATH ([ngspice.sourceforge.io](https://ngspice.sourceforge.io/))
- **Python 3.10+** with `numpy` and `matplotlib`
- **schemdraw** + **pillow** for circuit visualization (optional)
- **uv** recommended for running scripts (`uv run scripts/run_sim.py`)

## Quick Start

```bash
# Run a simulation and generate a Bode plot
uv run scripts/run_sim.py my_filter.cir --plot bode.png

# Parse a rawfile
uv run scripts/parse_rawfile.py output.raw
uv run scripts/parse_rawfile.py output.raw --csv > data.csv
uv run scripts/parse_rawfile.py output.raw --json > data.json
```

## What the Skill Covers

1. **Netlist syntax** — SPICE3 format, components, subcircuits, models, parameters
2. **Initial conditions & UIC** — `ic=` on components, `.tran UIC`, when and why to use it
3. **Analysis types** — `.ac`, `.dc`, `.tran`, `.op`, `.step`, `.meas`
4. **Binary rawfile parsing** — struct-level unpacking of ngspice's native format
5. **Monte Carlo analysis** — Python-driven tolerance sweeps with component tolerance tables
6. **Temperature sweeps** — Manual TC application for passives + `.step temp` for semiconductors
7. **Measurement extraction** — `.meas` directives + stdout parsing
8. **Plotting** — Bode plots, transient waveforms
9. **Quick reference** — Common elements (switches, coupled inductors, behavioral sources), source functions (PULSE, SIN, EXP, PWL)
10. **Common pitfalls** — Convergence, node naming, value suffixes, UIC gotchas
11. **Circuit visualization** — schemdraw for quick PNGs, KiCad export for interactive editing

## Compatible Agents

- GitHub Copilot (CLI, VS Code, JetBrains)
- Claude Code / Claude.ai
- OpenAI Codex
- Any agent supporting the SKILL.md convention

## License

MIT
