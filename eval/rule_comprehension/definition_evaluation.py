import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.replicate import ReplicateMultiModal
from llama_index.core.indices import VectorStoreIndex
from llama_index.multi_modal_llms.replicate.base import REPLICATE_MULTI_MODAL_LLM_MODELS
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
import pandas as pd
from tqdm import tqdm
from metrics.metrics import eval_definition_qa
from openai import OpenAI
import base64


def load_output_csv(model, overwrite_answers=False):
    # if output csv does not exist, create it
    csv_name = f"definition_evaluation_{model}.csv"
    if not os.path.exists(csv_name) or overwrite_answers:
        questions_pd = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "dataset",
                "rule_comprehension",
                "rule_definition_qa.csv",
            )
        )
        questions_pd.to_csv(csv_name, index=False)
    else:
        questions_pd = pd.read_csv(csv_name)
    return questions_pd, csv_name


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_qwen_vlm(question, image_path, base_url, api_key):
    if not base_url or not api_key:
        raise ValueError(
            "QWEN_BASE_URL and QWEN_API_KEY environment variables must be set "
            "to use qwen-3.5-27b-fp8 model"
        )
    client = OpenAI(base_url=base_url, api_key=api_key)
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="Qwen/Qwen3.5-27B-FP8",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=500,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content or ""


def run_thread(model, question, image_path):
    if model == "llava-13b":
        # API token of the model/pipeline that we will be using
        REPLICATE_API_TOKEN = ""
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
        model = REPLICATE_MULTI_MODAL_LLM_MODELS["llava-13b"]
        multi_modal_llm = ReplicateMultiModal(model=model, max_new_tokens=100)
    elif (
        model == "gpt-4-1106-vision-preview"
        or model == "gpt-4-1106-vision-preview+context"
    ):
        # OpenAI model
        multi_modal_llm = OpenAIMultiModal(
            model="gpt-4-vision-preview", max_new_tokens=100
        )
    elif model in ["qwen-3.5-27b-fp8", "qwen-3.5-27b-fp8+context"]:
        # Qwen VL model via OpenAI-compatible API - use native OpenAI client
        return call_qwen_vlm(
            question, image_path, os.getenv("QWEN_BASE_URL"), os.getenv("QWEN_API_KEY")
        )
    else:
        raise ValueError("Invalid model")

    # load question image
    image_document = SimpleDirectoryReader(input_files=[image_path]).load_data()

    # get response from model
    rag_response = multi_modal_llm.complete(
        prompt=question, image_documents=image_document
    )
    return str(rag_response)


def save_results(model, macro_avg, definitions_avg, multi_avg, single_avg, all_answers):
    print(f"\nMacro avg: {macro_avg}")
    print(f"\nDefinitions: {definitions_avg}")
    print(f"\nMulti avg: {multi_avg}")
    print(f"\nSingle avg: {single_avg}")
    print(f"\nAll answers: {all_answers}")

    # Save results to txt file
    with open(f"definition_evaluation_{model}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg}")
        text_file.write(f"\nDefinitions: {definitions_avg}")
        text_file.write(f"\nMulti avg: {multi_avg}")
        text_file.write(f"\nSingle avg: {single_avg}")
        text_file.write(f"\nAll answers: {all_answers}")


def retrieve_context(question):
    # load all context from original text document
    txt_path = "../../dataset/docs/rules_pdfplumber1.txt"
    context = open(txt_path, "r", encoding="utf-8").read()

    question_with_context = (
        question[:80]
        + f"Below is context from the FSAE rule document which might or might not "
        f"be relevant for the question: \n\n```\n{context}\n```\n\n" + question[117:]
    )

    return question_with_context


def run_inference(model, overwrite_answers=False):
    questions_pd, csv_name = load_output_csv(model, overwrite_answers)

    for i, row in tqdm(
        questions_pd.iterrows(),
        total=len(questions_pd),
        desc=f"generating responses for {model}",
    ):
        try:
            model_prediction = row["model_prediction"]
        except KeyError:
            model_prediction = None
        if not pd.isnull(model_prediction) and not overwrite_answers:
            continue

        question = row["question"]
        image_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "dataset",
            "rule_comprehension",
            "rule_definition_qa",
            row["image"],
        )

        if model == "gpt-4-1106-vision-preview+context":
            question = retrieve_context(question)
        response = run_thread(model, question, image_path)

        questions_pd.at[i, "model_prediction"] = response
        questions_pd.to_csv(csv_name, index=False)

    macro_avg, definitions_avg, multi_avg, single_avg, all_answers = eval_definition_qa(
        csv_name
    )
    save_results(model, macro_avg, definitions_avg, multi_avg, single_avg, all_answers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Definition evaluation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., qwen-3.5-27b-fp8, gpt-4-1106-vision-preview)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictions",
    )
    args = parser.parse_args()
    run_inference(args.model, args.overwrite)
