# Development Guide

## Development Workflow

### Initial Setup

```bash
# Install dependencies
pixi install

# Activate the dev environment (or use direnv for automatic activation)
pixi shell -e dev
```

### Common Tasks

```bash
# pre-commit check
pixi run pre-commit

# Run tests
pixi run test

# Run all checks (format, lint, types, test)
pixi run all
```

### Adding Dependencies

```bash
# Add a conda dependency
pixi add <package-name>

# Add a PyPI dependency
pixi add --pypi <package-name>

# Add a dev dependency
pixi add --dev <package-name>
```
## Environment Setup

The project uses `.envrc` for automatic environment activation via direnv.

### What is `.envrc`?

The `.envrc` file is used by direnv to automatically load environment variables when you enter the project directory.

**Contents:**
- `watch_file pixi.lock` - Tells direnv to reload when `pixi.lock` changes (e.g., after adding dependencies)
- `eval "$(pixi shell-hook -e dev)"` - Activates the pixi dev environment, setting up the Python environment, PATH, and other variables

**How it works:**
- When you `cd` into the project directory, direnv automatically activates the pixi dev environment
- When you leave, it deactivates
- No need to manually run `pixi shell` or activate virtual environments

**Setup required:**
- Install `direnv` and add it to your shell (usually a one-time setup)
- Run `direnv allow` once in the project directory to trust the `.envrc` file

This keeps your development environment activated automatically as you work.
