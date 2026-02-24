# Contributing

See the full [CONTRIBUTING.md](https://github.com/cloudQuant/fincore/blob/main/CONTRIBUTING.md) for details.

## Quick Summary

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run `pytest tests/` and `ruff check fincore/ tests/`
5. Submit a Pull Request

## Coding Conventions

- **Python 3.11+** with modern syntax
- **120 char** line length (Ruff enforced)
- **NumPy-style** docstrings on all public functions
- **Explicit imports** — no star imports
- **Lazy loading** for heavy dependencies
