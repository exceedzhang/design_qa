import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from metrics.metrics import (
    eval_retrieval_qa,
    eval_compilation_qa,
    eval_definition_qa,
    eval_presence_qa,
    eval_dimensions_qa,
    eval_functional_performance_qa,
)
from rule_extraction.extraction_evaluation import run_inference as run_extraction
from rule_comprehension.definition_evaluation import run_inference as run_definition
from rule_comprehension.presence_evaluation import run_inference as run_presence
from rule_compliance.dimension_evaluation import run_inference as run_dimension
from rule_compliance.functional_performance_evaluation import (
    run_inference as run_functional,
)


SUPPORTED_MODELS = [
    "qwen-3.5-27b-fp8",
    "qwen-3.5-27b-fp8+RAG",
    "gpt-4-0125-preview",
    "gpt-4-0125-preview+RAG",
    "gpt-4-1106-vision-preview",
    "gpt-4-1106-vision-preview+RAG",
    "llama-2-70b-chat",
    "llava-13b",
]


def run_inference_parallel(model, overwrite, max_workers=5):
    """Run all inference tasks in parallel using ThreadPoolExecutor."""

    if "RAG" in model and not os.path.exists("index"):
        print("Creating RAG index before parallel execution...")
        from eval.rule_compliance.dimension_evaluation import create_index

        index = create_index()
        index.storage_context.persist("index")
        print("RAG index created.")

    tasks = [
        ("extraction (retrieval + compilation)", run_extraction),
        ("definition", run_definition),
        ("presence", run_presence),
        ("dimension", run_dimension),
        ("functional performance", run_functional),
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(fn, model, overwrite): name for name, fn in tasks
        }

        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                future.result()
                print(f"[OK] {task_name} completed")
            except Exception as e:
                print(f"[FAIL] {task_name} failed: {e}")


def get_csv_path(task, model, question_type=None):
    if task == "retrieval":
        return f"retrieval_evaluation_{model}.csv"
    elif task == "compilation":
        return f"compilation_evaluation_{model}.csv"
    elif task == "definition":
        return f"definition_evaluation_{model}.csv"
    elif task == "presence":
        return f"presence_evaluation_{model}.csv"
    elif task == "dimension":
        return f"dimension_{question_type}_evaluation_{model}.csv"
    elif task == "functional_performance":
        return f"dimension_functional_performance_evaluation_{model}.csv"
    else:
        raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser(
        description="Full DesignQA evaluation. Provide --model to run inference + scoring, "
        "or provide CSV paths for scoring-only mode."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model to evaluate. Supported: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictions when running inference",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference step and only score existing CSVs",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run inference tasks in parallel (default: sequential)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of parallel workers for inference (default: 5)",
    )
    parser.add_argument(
        "--path_to_retrieval",
        type=str,
        default=None,
        help="Path to CSV for retrieval (overrides auto-generated path)",
    )
    parser.add_argument(
        "--path_to_compilation",
        type=str,
        default=None,
        help="Path to CSV for compilation (overrides auto-generated path)",
    )
    parser.add_argument(
        "--path_to_dimension",
        type=str,
        default=None,
        help="Path to CSV for dimension (overrides auto-generated path)",
    )
    parser.add_argument(
        "--path_to_functional_performance",
        type=str,
        default=None,
        help="Path to CSV for functional performance (overrides auto-generated path)",
    )
    parser.add_argument(
        "--path_to_definition",
        type=str,
        default=None,
        help="Path to CSV for definition (overrides auto-generated path)",
    )
    parser.add_argument(
        "--path_to_presence",
        type=str,
        default=None,
        help="Path to CSV for presence (overrides auto-generated path)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="results.txt",
        help="Path to save evaluation results (default: results.txt)",
    )

    args = parser.parse_args()

    if args.model and args.model not in SUPPORTED_MODELS:
        print(
            f"Warning: model '{args.model}' not in known list. Proceeding anyway.\n"
            f"Supported: {', '.join(SUPPORTED_MODELS)}"
        )

    retrieval_csv = args.path_to_retrieval or (
        get_csv_path("retrieval", args.model) if args.model else None
    )
    compilation_csv = args.path_to_compilation or (
        get_csv_path("compilation", args.model) if args.model else None
    )
    definition_csv = args.path_to_definition or (
        get_csv_path("definition", args.model) if args.model else None
    )
    presence_csv = args.path_to_presence or (
        get_csv_path("presence", args.model) if args.model else None
    )
    dimension_csv = args.path_to_dimension or (
        get_csv_path("dimension", args.model, "context") if args.model else None
    )
    dimension_detailed_csv = (
        get_csv_path("dimension", args.model, "detailed_context")
        if args.model
        else None
    )
    functional_csv = args.path_to_functional_performance or (
        get_csv_path("functional_performance", args.model) if args.model else None
    )

    if args.model and not args.skip_inference:
        print(f"\n=== Running inference with model: {args.model} ===\n")

        if args.parallel:
            print(
                f"Running inference in parallel (max_workers={args.max_workers})...\n"
            )
            run_inference_parallel(args.model, args.overwrite, args.max_workers)
        else:
            print("Extraction (retrieval + compilation)...")
            run_extraction(args.model, args.overwrite)

            print("Definition evaluation...")
            run_definition(args.model, args.overwrite)

            print("Presence evaluation...")
            run_presence(args.model, args.overwrite)

            print("Dimension evaluation...")
            run_dimension(args.model, args.overwrite)

            print("Functional performance evaluation...")
            run_functional(args.model, args.overwrite)

    print("\n=== Scoring results ===\n")

    all_subsets = []

    macro_avg_retrieval = None
    macro_avg_compilation = None
    macro_avg_definition = None
    macro_avg_presence = None
    macro_avg_accuracy_dimension = None
    macro_avg_accuracy_dimension_detailed = None
    macro_avg_accuracy_functional = None
    all_answers_retrieval = None
    all_answers_compilation = None
    definitions_qs_definition_avg = None
    multi_qs_definition_avg = None
    single_qs_definition_avg = None
    all_answers_definition = None
    definitions_qs_presence_avg = None
    multi_qs_presence_avg = None
    single_qs_presence_avg = None
    all_answers_presence = None
    direct_dim_avg = None
    scale_bar_avg = None
    all_accuracies_dimension = None
    macro_avg_bleus_dimension = None
    all_bleus_dimension = None
    macro_avg_rogues_dimension = None
    all_rogues_dimension = None
    direct_dim_avg_detailed = None
    scale_bar_avg_detailed = None
    all_accuracies_dimension_detailed = None
    macro_avg_bleus_dimension_detailed = None
    all_bleus_dimension_detailed = None
    macro_avg_rogues_dimension_detailed = None
    all_rogues_dimension_detailed = None
    all_accuracies_functional = None
    macro_avg_bleus_functional = None
    all_bleus_functional = None
    macro_avg_rogues_functional = None
    all_rogues_functional = None

    if retrieval_csv:
        print(f"Scoring retrieval: {retrieval_csv}")
        macro_avg_retrieval, all_answers_retrieval = eval_retrieval_qa(retrieval_csv)
        all_subsets.append(macro_avg_retrieval)

    if compilation_csv:
        print(f"Scoring compilation: {compilation_csv}")
        macro_avg_compilation, all_answers_compilation = eval_compilation_qa(
            compilation_csv
        )
        all_subsets.append(macro_avg_compilation)

    if definition_csv:
        print(f"Scoring definition: {definition_csv}")
        (
            macro_avg_definition,
            definitions_qs_definition_avg,
            multi_qs_definition_avg,
            single_qs_definition_avg,
            all_answers_definition,
        ) = eval_definition_qa(definition_csv)
        all_subsets.append(macro_avg_definition)

    if presence_csv:
        print(f"Scoring presence: {presence_csv}")
        (
            macro_avg_presence,
            definitions_qs_presence_avg,
            multi_qs_presence_avg,
            single_qs_presence_avg,
            all_answers_presence,
        ) = eval_presence_qa(presence_csv)
        all_subsets.append(macro_avg_presence)

    if dimension_csv:
        print(f"Scoring dimension: {dimension_csv}")
        (
            macro_avg_accuracy_dimension,
            direct_dim_avg,
            scale_bar_avg,
            all_accuracies_dimension,
            macro_avg_bleus_dimension,
            all_bleus_dimension,
            macro_avg_rogues_dimension,
            all_rogues_dimension,
        ) = eval_dimensions_qa(dimension_csv)

    if dimension_detailed_csv:
        print(f"Scoring dimension (detailed_context): {dimension_detailed_csv}")
        (
            macro_avg_accuracy_dimension_detailed,
            direct_dim_avg_detailed,
            scale_bar_avg_detailed,
            all_accuracies_dimension_detailed,
            macro_avg_bleus_dimension_detailed,
            all_bleus_dimension_detailed,
            macro_avg_rogues_dimension_detailed,
            all_rogues_dimension_detailed,
        ) = eval_dimensions_qa(dimension_detailed_csv)

    if dimension_csv or dimension_detailed_csv:
        dim_scores = [
            s
            for s in [
                macro_avg_accuracy_dimension,
                macro_avg_accuracy_dimension_detailed,
            ]
            if s is not None
        ]
        if dim_scores:
            all_subsets.append(sum(dim_scores) / len(dim_scores))

    if functional_csv:
        print(f"Scoring functional performance: {functional_csv}")
        (
            macro_avg_accuracy_functional,
            all_accuracies_functional,
            macro_avg_bleus_functional,
            all_bleus_functional,
            macro_avg_rogues_functional,
            all_rogues_functional,
        ) = eval_functional_performance_qa(functional_csv)
        all_subsets.append(macro_avg_accuracy_functional)

    if os.path.exists(args.save_path):
        response = (
            input(f"File '{args.save_path}' already exists. Overwrite? (y/n): ")
            .lower()
            .strip()
        )
        if response not in ["y", "yes"]:
            print("Cancelled.")
            return

    num_subsets = len(all_subsets)
    overall_score = sum(all_subsets) / num_subsets if num_subsets > 0 else 0

    with open(args.save_path, "w") as f:
        f.write("DESIGNQA EVALUATION RESULTS:\n")
        f.write("-*-" * 20 + "\n")
        f.write("-*-" * 20 + "\n")
        f.write(f"OVERALL SCORE: {overall_score}\n")
        f.write(f"Model: {args.model or 'N/A'}\n")
        f.write(f"Subsets scored: {num_subsets}/6\n")
        f.write("-*-" * 20 + "\n")
        f.write(f"Retrieval Score (Avg F1 BoW): {macro_avg_retrieval}\n")
        f.write(f"Compilation Score (Avg F1 Rules): {macro_avg_compilation}\n")
        f.write(f"Definition Score (Avg F1 BoC): {macro_avg_definition}\n")
        f.write(f"Presence Score (Avg Accuracy): {macro_avg_presence}\n")
        f.write(f"Dimension Score (Avg Accuracy): {macro_avg_accuracy_dimension}\n")
        f.write(
            f"Functional Performance Score (Avg Accuracy): {macro_avg_accuracy_functional}\n"
        )
        f.write("-*-" * 20 + "\n\n\n")
        f.write("Below scores by subset are provided for diagnostic purposes:\n")
        f.write("---" * 20 + "\n")
        f.write("RETRIEVAL\n---\n")
        if all_answers_retrieval:
            f.write(f"All F1 BoWs:\n{all_answers_retrieval}\n")
        else:
            f.write("No data.\n")

        f.write("---" * 20 + "\n")
        f.write("COMPILATION\n---\n")
        if all_answers_compilation:
            f.write(f"All F1 Rules:\n{all_answers_compilation}\n")
        else:
            f.write("No data.\n")

        f.write("---" * 20 + "\n")
        f.write("DEFINITION\n---\n")
        if all_answers_definition:
            f.write(
                f"Avg F1 BoC on definition-components:\n{definitions_qs_definition_avg}\n"
            )
            f.write(
                f"Avg F1 BoC on multimention-components:\n{multi_qs_definition_avg}\n"
            )
            f.write(
                f"Avg F1 BoC on no-mention-components:\n{single_qs_definition_avg}\n"
            )
            f.write(f"All F1 BoC:\n{all_answers_definition}\n")
        else:
            f.write("No data.\n")

        f.write("---" * 20 + "\n")
        f.write("PRESENCE\n---\n")
        if all_answers_presence:
            f.write(
                f"Avg accuracy on definition-components:\n{definitions_qs_presence_avg}\n"
            )
            f.write(
                f"Avg accuracy on multimention-components:\n{multi_qs_presence_avg}\n"
            )
            f.write(
                f"Avg accuracy on no-mention-components:\n{single_qs_presence_avg}\n"
            )
            f.write(f"All accuracies:\n{all_answers_presence}\n")
        else:
            f.write("No data.\n")

        f.write("---" * 20 + "\n")
        f.write("DIMENSION\n---\n")
        if all_accuracies_dimension:
            f.write(f"[context]\n")
            f.write(f"Avg accuracy directly-dimensioned:\n{direct_dim_avg}\n")
            f.write(f"Avg accuracy scale-bar-dimensioned:\n{scale_bar_avg}\n")
            f.write(f"All accuracies:\n{all_accuracies_dimension}\n")
            f.write(f"Avg BLEU score:\n{macro_avg_bleus_dimension}\n")
            f.write(f"All BLEU scores:\n{all_bleus_dimension}\n")
            f.write(f"Avg ROUGE score:\n{macro_avg_rogues_dimension}\n")
            f.write(f"All ROUGE scores:\n{all_rogues_dimension}\n")
        if all_accuracies_dimension_detailed:
            f.write(f"[detailed_context]\n")
            f.write(f"Avg accuracy directly-dimensioned:\n{direct_dim_avg_detailed}\n")
            f.write(f"Avg accuracy scale-bar-dimensioned:\n{scale_bar_avg_detailed}\n")
            f.write(f"All accuracies:\n{all_accuracies_dimension_detailed}\n")
            f.write(f"Avg BLEU score:\n{macro_avg_bleus_dimension_detailed}\n")
            f.write(f"All BLEU scores:\n{all_bleus_dimension_detailed}\n")
            f.write(f"Avg ROUGE score:\n{macro_avg_rogues_dimension_detailed}\n")
            f.write(f"All ROUGE scores:\n{all_rogues_dimension_detailed}\n")
        if not all_accuracies_dimension and not all_accuracies_dimension_detailed:
            f.write("No data.\n")

        f.write("---" * 20 + "\n")
        f.write("FUNCTIONAL PERFORMANCE\n---\n")
        if all_accuracies_functional:
            f.write(f"All accuracies:\n{all_accuracies_functional}\n")
            f.write(f"Avg BLEU score:\n{macro_avg_bleus_functional}\n")
            f.write(f"All BLEU scores:\n{all_bleus_functional}\n")
            f.write(f"Avg ROUGE score:\n{macro_avg_rogues_functional}\n")
            f.write(f"All ROUGE scores:\n{all_rogues_functional}\n")
        else:
            f.write("No data.\n")

    print(f"\nResults saved to {args.save_path}")
    print(f"Overall score: {overall_score}")


if __name__ == "__main__":
    main()
