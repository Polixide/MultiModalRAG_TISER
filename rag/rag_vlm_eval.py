import json

from PIL import Image
from tqdm.auto import tqdm

from eval.text_metrics import extract_answer_from_text, exact_match, f1_score
from modeling.collator_qwen import resize_pil
from config import RAG_TOP1_DATASET_FILE


SYSTEM_MESSAGE = (
    "You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection "
    "to answer queries about charts. Follow these steps: "
    "Step 1. Reason through the visual data step by step within the reasoning tags. "
    "Step 2. Given your previous reasoning, identify relevant temporal events in the given "
    "context for answering the given question within timeline tags. Assume relations in the "
    "context are unidirectional. "
    "Step 3. Reflect on your reasoning and the timeline to check for any errors or improvements "
    "within the reflection tags. "
    "Step 4. Make any necessary adjustments based on your reflection. "
    "Step 5. Provide your final, concise answer within the answer tags. "
    "If the answer is a number, just output the number nothing else. Otherwise output the entity or event, "
    "without any additional comments. "
    "Important: The reasoning, reflection and timeline sections are for your internal reasoning process. "
    "All the reflection and the timeline have to be contained inside the thinking section. "
    "Do not use enumerations or lists when writing, use plain text instead such as paragraphs. "
    "The response to the query must be entirely contained within the answer tags."
)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def run_inference_model(model, proc, example, max_new_tokens: int = 1024) -> str:
    question = example["question"]
    image = resize_pil(example["image"])

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f"Question: {question}\n\n"
                        "Temporal context: The provided chart contains the temporal context for this question. "
                        "Important: Use the chart to reason about the order, overlap and duration of events, "
                        "and answer exactly what is asked in the question. When the question asks about a "
                        "specific date or date range, identify the event whose interval actually includes that "
                        "date or fully covers that range. If the chart does not provide enough information to "
                        "answer, answer Unknown."
                    ),
                },
            ],
        },
    ]

    inputs = proc.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return proc.decode(gen, skip_special_tokens=True)


def eval_rag_plus_vlm(
    dataset_file: str = RAG_TOP1_DATASET_FILE,
    model=None,
    processor=None,
):
    em_list = []
    f1_list = []

    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Eval RAG VLM"):
            data = json.loads(line)
            q = data.get("question")
            gold = data.get("gold_answer")
            imgpath = data.get("top1_path")

            if not imgpath:
                continue

            image = load_image(imgpath)
            example = {"question": q, "image": image}
            pred_full = run_inference_model(model, processor, example)
            pred = extract_answer_from_text(pred_full)

            em_list.append(exact_match(pred, gold))
            f1_list.append(f1_score(pred, gold))

    em = sum(em_list) / len(em_list) if em_list else 0.0
    f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    return {"em_post_rag": em, "f1_post_rag": f1}
