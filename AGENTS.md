# AGENTS.md - Agent Coding Guidelines for DesignQA

## Project Overview

DesignQA is a benchmark for evaluating multimodal LLMs' understanding of engineering documentation (FSAE competition rules). It contains evaluation scripts in Python with no formal test suite.

## Build/Run Commands

### Setup
```bash
# Create and activate conda environment
conda create -n design_qa python=3.10 -y
conda activate design_qa
pip install -r requirements.txt
```

### Running Evaluations
```bash
# Full benchmark evaluation
python eval/full_evaluation.py \
  --path_to_retrieval <path_to_csv> \
  --path_to_compilation <path_to_csv> \
  --path_to_definition <path_to_csv> \
  --path_to_presence <path_to_csv> \
  --path_to_dimension <path_to_csv> \
  --path_to_functional_performance <path_to_csv>
```

### Running Single Evaluation Tasks
```bash
# Retrieval evaluation
python -c "from eval.metrics.metrics import eval_retrieval_qa; print(eval_retrieval_qa('your_file.csv'))"

# Compilation evaluation  
python -c "from eval.metrics.metrics import eval_compilation_qa; print(eval_compilation_qa('your_file.csv'))"

# Definition evaluation
python -c "from eval.metrics.metrics import eval_definition_qa; print(eval_definition_qa('your_file.csv'))"

# Presence evaluation
python -c "from eval.metrics.metrics import eval_presence_qa; print(eval_presence_qa('your_file.csv'))"

# Dimension evaluation
python -c "from eval.metrics.metrics import eval_dimensions_qa; print(eval_dimensions_qa('your_file.csv'))"

# Functional performance evaluation
python -c "from eval.metrics.metrics import eval_functional_performance_qa; print(eval_functional_performance_qa('your_file.csv'))"
```

### Testing Individual Metrics
```bash
# Run metrics file directly (contains test code in __main__)
python eval/metrics/metrics.py
```

## Code Style Guidelines

### General Conventions
- Python 3.10+
- Use `argparse` for CLI argument parsing
- Use `if __name__ == "__main__":` guard for executable scripts

### Naming Conventions
- **Functions/variables**: `snake_case` (e.g., `eval_retrieval_qa`, `f1_scores`)
- **Classes**: `PascalCase` (e.g., `VectorStoreIndex` from llama_index)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `REPLICATE_MULTI_MODAL_LLM_MODELS`)
- **Files**: `snake_case.py`

### Imports
- Standard library first, then third-party, then local
- Group by: builtins → external → project-specific
- Use explicit imports (no `from x import *`)
```python
import os
import csv
import pandas as pd
from tqdm import tqdm

from eval.metrics import eval_retrieval_qa, eval_compilation_qa
```

### Type Hints
- Not currently used in codebase, but recommended for new code
- Use `from typing import` for complex types

### Docstrings
- Use Google-style docstrings
```python
def eval_retrieval_qa(results_csv):
    """
    Evaluate retrieval QA predictions.

    Args:
        results_csv: Path to CSV with columns "model_prediction" and "ground_truth"

    Returns:
        Tuple of (macro-averaged F1 score, list of all F1 scores)
    """
```

### Error Handling
- Use try/except for operations that may fail (file I/O, API calls)
- Catch specific exceptions when possible
- Print user-friendly error messages
```python
try:
    response = llm.complete(question)
except Exception as e:
    print(f"Error: {e}")
    print(f"Question: {question}")
    response = ' '
```

### CSV Handling
- Use `pandas` for CSV operations
- Expected columns: `ground_truth`, `model_prediction`, optional `mentions`, `dimension_type`, `explanation`
- Use `pd.read_csv()` and `df.to_csv(index=False)`

### File Paths
- Use relative paths from project root
- Use `os.path.exists()` to check file existence
- Use `tqdm` for progress indicators on iterative operations

### Code Organization
- Evaluation metrics in `eval/metrics/metrics.py`
- Evaluation runners in `eval/rule_*/`
- Helper scripts in `scripts/`
- Entry points use `argparse` with descriptive help text

### Linting (Recommended)
Install and run:
```bash
pip install ruff
ruff check .
```

For automatic fixes:
```bash
ruff check --fix .
```

### Adding New Evaluation Metrics
1. Add function to `eval/metrics/metrics.py`
2. Follow naming: `eval_<task>_qa`
3. Return tuple: `(overall_score, per_question_scores)` or similar
4. Update `eval/full_evaluation.py` to support new metric

### CSV Format for Evaluation
Each evaluation CSV must have:
- `ground_truth`: Correct answer
- `model_prediction`: Model's answer

Optional columns:
- `mentions`: For definition/presence tasks ("definition", "mentioned", "not mentioned")
- `dimension_type`: For dimension tasks ("direct", "scale_bar")
- `explanation`: Ground truth explanation for scoring
- `question`: The original question (for debugging)
