"""Validate that submitted teaching notebooks contain genuine executions."""

from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "01_data_preparation_sampling_and_split.ipynb",
    "02_fast_lts_validation_and_scalability.ipynb",
    "03_training_eda_and_model_specification.ipynb",
    "04_clean_ols_and_lts_comparison.ipynb",
    "05_contamination_stress_test.ipynb",
]


def test_teaching_notebooks_are_fully_executed_without_errors() -> None:
    """Require an execution count and no error output in every code cell."""
    for filename in NOTEBOOKS:
        path = PROJECT_ROOT / "notebooks" / filename
        document = nbformat.read(path, as_version=4)
        code_cells = [cell for cell in document.cells if cell.cell_type == "code"]

        assert code_cells, f"{filename} has no code cells."
        assert all(cell.execution_count is not None for cell in code_cells), (
            f"{filename} contains unexecuted code."
        )
        assert not any(
            output.output_type == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        ), f"{filename} contains a stored execution error."
