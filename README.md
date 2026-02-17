# circuit-sim-skill

An [agent skill](https://docs.github.com/copilot/concepts/agents/about-agent-skills)
that teaches AI coding assistants how to drive **ngspice** for analog circuit
simulation — from netlist authoring through binary rawfile parsing to
publication-quality plots.

> **Schematic diagrams:** To convert a netlist into a circuit schematic, see
> [netlist-to-schematic-skill](https://github.com/faisal-shah/netlist-to-schematic-skill).

## What's Included

| File | Required | Purpose |
|------|----------|---------|
| `SKILL.md` | **yes** | Main skill file — loaded by the agent framework |
| `scripts/parse_rawfile.py` | **yes** | Binary rawfile parser (CLI + library) |
| `scripts/run_sim.py` | **yes** | End-to-end sim runner with .meas/.step/UIC handling |
| `README.md` | no | This file (repo documentation only) |
| `AGENTS.md` | no | AI context for developing the skill itself |
| `LICENSE` | no | MIT license text |
| `examples/` | no | Reference netlists for testing changes to the skill |
| `tags` | no | ctags file |

## Installation

### GitHub Copilot / VS Code

Clone and copy only the required files:

```bash
git clone https://github.com/faisal-shah/circuit-sim-skill.git
```

Then install into your project or user-level skills:

```bash
# Option 1: project-level
mkdir -p .github/skills/circuit-sim/scripts
cp circuit-sim-skill/SKILL.md .github/skills/circuit-sim/
cp circuit-sim-skill/scripts/*.py .github/skills/circuit-sim/scripts/

# Option 2: user-level (all projects)
mkdir -p ~/.copilot/skills/circuit-sim/scripts
cp circuit-sim-skill/SKILL.md ~/.copilot/skills/circuit-sim/
cp circuit-sim-skill/scripts/*.py ~/.copilot/skills/circuit-sim/scripts/
```

### Claude Code

```bash
mkdir -p ~/.claude/skills/circuit-sim/scripts
cp circuit-sim-skill/SKILL.md ~/.claude/skills/circuit-sim/
cp circuit-sim-skill/scripts/*.py ~/.claude/skills/circuit-sim/scripts/
```

### OpenAI Codex

```bash
mkdir -p ~/.codex/skills/circuit-sim/scripts
cp circuit-sim-skill/SKILL.md ~/.codex/skills/circuit-sim/
cp circuit-sim-skill/scripts/*.py ~/.codex/skills/circuit-sim/scripts/
```

## Prerequisites

- **ngspice** installed and on PATH ([ngspice.sourceforge.io](https://ngspice.sourceforge.io/))
- **Python 3.10+** with `numpy` and `matplotlib`
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

## Related Skills

For converting netlists into **circuit schematic diagrams**, see
[netlist-to-schematic-skill](https://github.com/faisal-shah/netlist-to-schematic-skill).

## Compatible Agents

- GitHub Copilot (CLI, VS Code, JetBrains)
- Claude Code / Claude.ai
- OpenAI Codex
- Any agent supporting the SKILL.md convention

## License

MIT
