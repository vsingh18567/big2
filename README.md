# Big 2 Simulator

A Python-based simulator for the Big 2 card game.

## Development Setup

This project uses modern Python tooling:
- `uv` for dependency management and virtual environments
- `ruff` for linting and formatting
- `mypy` for static type checking

### Prerequisites

Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

1. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
uv pip install ruff mypy hatchling typing-extensions
```

### Development Commands

- Run type checking:
```bash
mypy simulator
```

- Run linting:
```bash
ruff check simulator
```

- Run formatting:
```bash
ruff format simulator
```
