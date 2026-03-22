import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.replicate import ReplicateMultiModal
from llama_index.core.indices import VectorStoreIndex
from llama_index.multi_modal_llms.replicate.base import REPLICATE_MULTI_MODAL_LLM_MODELS
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import csv
import pandas as pd
from tqdm import tqdm
from metrics.metrics import eval_dimensions_qa
from openai import OpenAI
import base64


SCALE_BAR_INSTRUCTION = """
You are an engineering drawing analysis assistant specializing in scale bar measurements.

CRITICAL REQUIREMENTS:

1. READ THE SCALE BAR VALUE
   - Look at the dimension line at the TOP of the engineering drawing
   - The labeled number (e.g., "3202.4") represents the actual length in millimeters
   - You MUST identify this scale bar value before any other measurement

2. SHOW CALCULATION STEPS
   - Step 1: State the scale bar value you identified
   - Step 2: Estimate the pixel ratio (how many mm per pixel)
   - Step 3: Measure the target feature relative to the scale bar
   - Step 4: Calculate the actual dimension using the ratio
   - Step 5: Compare with the rule requirement and state compliance

3. RESPONSE FORMAT (MANDATORY)
   Your response MUST follow this exact format:
   
   Explanation: [Your step-by-step calculation and reasoning]
   Answer: [yes or no only]
   
   Do NOT provide any text after "Answer:" except "yes" or "no".

EXAMPLE:
Question: Rule V.1.2 requires minimum wheelbase of 1525mm.
Explanation: Step 1: Scale bar = 3202.4mm. Step 2: Wheelbase spans ~60% of total vehicle length. Step 3: 0.60 × 3202.4mm = 1921mm. Step 4: 1921mm > 1525mm. The design complies.
Answer: yes
"""

DIRECT_DIMENSION_INSTRUCTION = """
You are an engineering drawing analysis assistant specializing in reading explicit dimensions.

CRITICAL REQUIREMENTS:

1. READ EXPLICIT DIMENSIONS ONLY
   - Find dimension values that are DIRECTLY LABELED on the drawing
   - DO NOT estimate or calculate from scale bar
   - DO NOT guess dimensions that are not shown
   - When you see a dimension, VERIFY it points to the correct measurement target

2. UNDERSTAND THE MEASUREMENT METHOD
   - Some rules require CALCULATED values (e.g., "radius minus distance = ground clearance")
   - Some measurements need COMBINATION (e.g., "half-width × 2 = total track width")
   - When the rule specifies a derived measurement, you must perform the calculation
   - Read the rule requirement carefully to identify what value to calculate

3. IDENTIFY THE CORRECT VIEW
   - Top/Side/Front views measure different things
   - Some rules specify which view to use (e.g., "plan view", "side elevation")
   - Use ONLY the view specified in the rule for the measurement
   - If the drawing only shows one view, that is the intended view

4. VERIFY DIMENSION CORRESPONDENCE
   - Extension lines show what dimension the label refers to
   - A dimension labeled "24.30" near a tube does not necessarily mean tube diameter
   - Confirm the dimension label actually measures what you think it measures

5. RESPONSE FORMAT (MANDATORY)
   Your response MUST follow this exact format:
   
   Explanation: [State the labeled dimension(s), explain your calculation if needed, then your reasoning]
   Answer: [yes or no only]

EXAMPLE 1 - Simple Comparison:
Question: Rule V.1.2 requires minimum wheelbase of 1525mm.
Explanation: The drawing explicitly labels the wheelbase as 1524.9mm. Since 1524.9mm < 1525mm, the requirement is not met.
Answer: no

EXAMPLE 2 - Calculated Value:
Question: Rule V.1.4.2 requires ground clearance (tire radius minus distance to lower side impact) to be 75mm or less.
Explanation: The drawing shows: tire radius = R203.2mm, distance to lower side impact = 128mm. Calculated ground clearance = 203.2 - 128 = 75.2mm. Since 75.2mm > 75mm, the requirement is not met.
Answer: no

EXAMPLE 3 - Multi-part Measurement:
Question: Rule T.7.6.1 specifies front wing must be within the outside of front tires.
Explanation: The drawing shows dimension "16.44" from centerline to outer tire edge. This is half the track width, so total = 16.44 × 2 = 32.88mm. The wing extends beyond this width.
Answer: yes
"""


def get_text_prompts(text_query_path):
    # get prompt dataset
    # text prompt
    queries = []
    with open(text_query_path, mode="r") as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)
        for row in csv_reader:
            queries.append(row[0])
    return queries


def load_output_csv(model, question_type, overwrite_answers=False):
    # if output csv does not exist, create it
    csv_name = f"dimension_{question_type}_evaluation_{model}.csv"
    if not os.path.exists(csv_name) or overwrite_answers:
        questions_pd = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "dataset",
                "rule_compliance",
                "rule_dimension_qa",
                question_type,
                f"rule_dimension_qa_{question_type}.csv",
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

    if "use the scale bar" in question.lower():
        system_instruction = SCALE_BAR_INSTRUCTION
    else:
        system_instruction = DIRECT_DIMENSION_INSTRUCTION

    client = OpenAI(base_url=base_url, api_key=api_key)
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="Qwen/Qwen3.5-27B-FP8",
        messages=[
            {
                "role": "system",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        max_tokens=1500,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return response.choices[0].message.content or ""


def run_thread(model, question, image_path, context):
    if model == "llava-13b":
        # API token of the model/pipeline that we will be using
        REPLICATE_API_TOKEN = ""
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
        model = REPLICATE_MULTI_MODAL_LLM_MODELS["llava-13b"]
        multi_modal_llm = ReplicateMultiModal(model=model, max_new_tokens=100)
    elif model in ["gpt-4-1106-vision-preview", "gpt-4-1106-vision-preview+RAG"]:
        # OpenAI model
        multi_modal_llm = OpenAIMultiModal(
            model="gpt-4-vision-preview", max_new_tokens=1500
        )
    elif model in ["qwen-3.5-27b-fp8", "qwen-3.5-27b-fp8+RAG"]:
        # Qwen VL model via OpenAI-compatible API
        question = add_context_to_prompt(question, context)
        return call_qwen_vlm(
            question, image_path, os.getenv("QWEN_BASE_URL"), os.getenv("QWEN_API_KEY")
        )
    else:
        raise ValueError("Invalid model")

    # load question image
    image_document = SimpleDirectoryReader(input_files=[image_path]).load_data()

    # modify text prompt to include context
    question = add_context_to_prompt(question, context)

    # get response from model
    rag_response = multi_modal_llm.complete(
        prompt=question, image_documents=image_document
    )
    return str(rag_response)


def add_context_to_prompt(prompt, context):
    if isinstance(context, str):  # if context is a string, it is the entire document
        prompt_with_context = (
            prompt[:80]
            + f"Below is context from the FSAE rule document which might or might not "
            f"be relevant for the question: \n\n```\n{context}\n```\n\n" + prompt[117:]
        )
    else:
        # sort the context by page
        context = sorted(context, key=lambda x: int(x.metadata["page_label"]))

        # add the context to the prompt
        prompt_with_context = (
            prompt[:80]
            + "Below is context from the FSAE rule document which might or might not "
            "be relevant for the question: \n\n```\n"
        )
        for doc in context:
            prompt_with_context += f"{doc.text}\n"
        prompt_with_context += "```\n\n" + prompt[117:]

    return prompt_with_context


def create_index():
    pdf_path = "../../dataset/docs/FSAE_Rules_2024_V1.pdf"
    text_documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

    chunk_size = 250
    transformations = [SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)]
    embedding_model = OpenAIEmbedding(model="text-embedding-3-large")
    index = VectorStoreIndex.from_documents(
        text_documents, embed_model=embedding_model, transformations=transformations
    )

    index.storage_context.persist("index")
    return index


def retrieve_context(index, question, top_k=10):
    if top_k == 0:
        # load all context from original text document
        txt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "dataset",
            "docs",
            "rules_pdfplumber1.txt",
        )
        context = open(txt_path, "r", encoding="utf-8").read()
    else:
        retriever = index.as_retriever(similarity_top_k=top_k)
        context = retriever.retrieve(question)
    return context


def save_results(
    model,
    question_type,
    macro_avg_accuracy,
    direct_dim_avg,
    scale_bar_avg,
    all_accuracies,
    macro_avg_bleus,
    all_bleus,
    macro_avg_rogues,
    all_rogues,
):
    print(f"Model: {model}")
    print(f"\nMacro avg: {macro_avg_accuracy}")
    print(f"\nDirect Dimension avg: {direct_dim_avg}")
    print(f"\nScale Bar avg: {scale_bar_avg}")
    print(f"\nAll accuracies: {all_accuracies}")
    print(f"\nMacro avg bleus: {macro_avg_bleus}")
    print(f"\nAll bleus: {all_bleus}")
    print(f"\nMacro avg rogues: {macro_avg_rogues}")
    print(f"\nAll rogues: {all_rogues}")

    with open(f"dimension_{question_type}_evaluation_{model}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg_accuracy}")
        text_file.write(f"\nDirect Dimension avg: {direct_dim_avg}")
        text_file.write(f"\nScale Bar avg: {scale_bar_avg}")
        text_file.write(f"\nAll accuracies: {all_accuracies}")
        text_file.write(f"\nMacro avg bleus: {macro_avg_bleus}")
        text_file.write(f"\nAll bleus: {all_bleus}")
        text_file.write(f"\nMacro avg rogues: {macro_avg_rogues}")
        text_file.write(f"\nAll rogues: {all_rogues}")


def run_inference(model, overwrite_answers=False):
    index = None
    if "RAG" in model and index is None:
        if os.path.exists("index"):
            print("Loading index from storage...")
            storage_context = StorageContext.from_defaults(persist_dir="index")
            index = load_index_from_storage(
                storage_context,
                embed_model=OpenAIEmbedding(model="text-embedding-3-large"),
            )
        else:
            print("Creating index...")
            index = create_index()
            index.storage_context.persist("index")
    for question_type in ["context", "detailed_context"]:
        questions_pd, csv_name = load_output_csv(
            model, question_type, overwrite_answers
        )

        for i, row in tqdm(
            questions_pd.iterrows(),
            total=len(questions_pd),
            desc=f"generating responses for {question_type} with {model}",
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
                "rule_compliance",
                "rule_dimension_qa",
                question_type,
                row["image"],
            )

            if model in ["gpt-4-1106-vision-preview+RAG", "llava-13b"]:
                context = retrieve_context(index, question, top_k=12)
            elif model in ["gpt-4-1106-vision-preview"]:
                context = retrieve_context(index, question, top_k=0)
            elif model in ["qwen-3.5-27b-fp8", "qwen-3.5-27b-fp8+RAG"]:
                context = retrieve_context(None, question, top_k=0)
            else:
                raise ValueError("Invalid model")

            try:
                response = run_thread(model, question, image_path, context)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Question: {question}")
                print(f"Index: {i}")
                response = " "

            questions_pd.at[i, "model_prediction"] = response
            questions_pd.to_csv(csv_name, index=False)

        (
            macro_avg_accuracy,
            direct_dim_avg,
            scale_bar_avg,
            all_accuracies,
            macro_avg_bleus,
            all_bleus,
            macro_avg_rogues,
            all_rogues,
        ) = eval_dimensions_qa(csv_name)

        save_results(
            model,
            question_type,
            macro_avg_accuracy,
            direct_dim_avg,
            scale_bar_avg,
            all_accuracies,
            macro_avg_bleus,
            all_bleus,
            macro_avg_rogues,
            all_rogues,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dimension evaluation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., qwen-3.5-27b-fp8, gpt-4-1106-vision-preview+RAG)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictions",
    )
    args = parser.parse_args()
    run_inference(args.model, args.overwrite)
