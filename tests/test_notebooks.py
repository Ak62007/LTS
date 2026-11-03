"""Validate code-only notebooks and their saved executions."""

import io
import tokenize
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "01_data_preparation_sampling_and_split.ipynb",
    "02_fast_lts_validation_and_scalability.ipynb",
    "03_training_eda_and_model_specification.ipynb",
    "04_clean_ols_and_lts_comparison.ipynb",
    "05_contamination_stress_test.ipynb",
    "06_final_report_execution_evidence.ipynb",
]


def test_notebooks_are_code_only_and_fully_executed_without_errors() -> None:
    """Require an execution count and no error output in every code cell."""
    for filename in NOTEBOOKS:
        path = PROJECT_ROOT / "notebooks" / filename
        document = nbformat.read(path, as_version=4)
        code_cells = [cell for cell in document.cells if cell.cell_type == "code"]

        assert code_cells, f"{filename} has no code cells."
        assert len(code_cells) == len(document.cells), filename
        assert not any(
            token.type == tokenize.COMMENT
            for cell in code_cells
            for token in tokenize.generate_tokens(io.StringIO(cell.source).readline)
        ), f"{filename} contains code comments."
        assert all(cell.execution_count is not None for cell in code_cells), (
            f"{filename} contains unexecuted code."
        )
        assert not any(
            output.output_type == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        ), f"{filename} contains a stored execution error."
