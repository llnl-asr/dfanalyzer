# Contributing to DFAnalyzer

First off, thank you for considering contributing to DFAnalyzer! We welcome any help to make this tool better.

## How Can I Contribute?

### Reporting Bugs

If you encounter a bug, please report it by opening an issue on our [GitHub Issue Tracker](https://github.com/llnl-asr/dfanalyzer/issues).

When reporting a bug, please include:

- A clear and descriptive title.
- Steps to reproduce the bug.
- What you expected to happen.
- What actually happened.
- Your environment details (e.g., OS, Python version, DFAnalyzer version, relevant libraries).

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, please open an issue on our [GitHub Issue Tracker](https://github.com/llnl-asr/dfanalyzer/issues) to discuss it.

Please include:

- A clear and descriptive title.
- A detailed description of the proposed enhancement.
- The use case or problem this enhancement would solve.
- Any potential benefits or drawbacks.

### Submitting Pull Requests

We love pull requests! If you'd like to contribute code:

1.  **Fork the Repository:** Start by forking the [main DFAnalyzer repository](https://github.com/llnl-asr/dfanalyzer).
2.  **Create a Branch:** Create a new branch in your fork for your changes (e.g., `git checkout -b feature/my-new-feature` or `bugfix/issue-123`).
3.  **Make Your Changes:** Implement your feature or bug fix.
4.  **Test Your Changes:** Ensure your changes pass all tests. You can run tests locally (see "Setting Up a Development Environment" below and refer to the test execution steps in `.github/workflows/ci.yml`).
5.  **Format and Lint:** Ensure your code adheres to our coding standards (see below).
6.  **Commit Your Changes:** Use clear and descriptive commit messages.
7.  **Push to Your Fork:** Push your changes to your forked repository.
8.  **Submit a Pull Request:** Open a pull request from your branch to the `main` branch of the DFAnalyzer repository. Provide a clear description of your changes in the PR.

## Coding Standards

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Please ensure your contributions adhere to these standards. Key configurations (from `pyproject.toml`):

- Line length: 88 characters
- Quote style: Preserve existing quotes (`quote-style = "preserve"`)

You can run Ruff locally to check and format your code:

```bash
ruff check .
ruff format .
```

## Setting Up a Development Environment

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/llnl-asr/dfanalyzer.git
    cd dfanalyzer
    ```

2.  **Install Python Dependencies:**
    Install DFAnalyzer in editable mode along with the dependencies required for running tests.

    ```bash
    python -m pip install --upgrade pip
    # Install DFAnalyzer in editable mode
    pip install -e .
    # Install dependencies for tests
    pip install -r tests/requirements.txt
    # Ensure build tools are available
    # pip install setuptools wheel
    ```

DFAnalyzer is a pure Python package, so no system build dependencies are required beyond Python itself.


We look forward to your contributions!
