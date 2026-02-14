import os

import pandas as pd
from datasets import Dataset, Image as HFImage
from sklearn.model_selection import train_test_split

from config import MAX_TARGET_CHARS


SYSTEM_MESSAGE = (
    "You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection "
    "to answer queries about charts.\n"
    "Follow these steps:\n"
    "Step 1. Reason through the visual data step by step within the <reasoning> tags.\n"
    "Step 2. Given your previous reasoning, identify relevant temporal events in the given "
    "context for answering the given question within <timeline> tags. Assume relations in the "
    "context are unidirectional.\n"
    "Step 3. Reflect on your reasoning and the timeline to check for any errors or improvements "
    "within the <reflection> tags.\n"
    "Step 4. Make any necessary adjustments based on your reflection. If there is additional "
    "reasoning required, go back to Step 1 (reason through the visual data step-by-step), "
    "otherwise move to the next step (Step 5).\n"
    "Step 5. Provide your final, concise answer within the <answer> tags. "
    "If the answer is a number, just output the number, nothing else. "
    "Otherwise output the entity or event, without any additional comments.\n"
    "Important: The <reasoning>, <reflection> and <timeline> sections are for your internal reasoning "
    "process. All the reflection and the timeline have to be contained inside the thinking section.\n"
    "Do not use enumerations or lists when writing, use plain text instead such as paragraphs.\n"
    "The response to the query must be entirely contained within the <answer> tags.\n"
    "Use the following format for your response:\n\n<reasoning>\n[Your step-by-step reasoning goes here. This is your internal thought process.]\n<timeline>\n[Relevant temporal events for answering the given question.]\n</timeline>\n<reflection>\n[Your reflection on your reasoning, checking for errors or improvements]\n</reflection>\n[Any adjustments to your thinking based on your reflection]\n</reasoning>\n<answer>\n[Your final, concise answer to the query.]\n</answer>\n"
    "When answering, always follow these rules:\n- Use the chart to reason about the order, overlap, and duration of events, and answer exactly what is asked in the question.\n- Identify the event or interval that actually covers the requested date or date range on the timeline.\n- If the requested period is a range (e.g. 2006-2007), the correct event must cover the whole range, not just its start or end.\n- If no event covers the requested date or the whole requested range, answer 'Unknown' (or the event labeled as Unknown in the chart).\n- Never pick an event only because it is the last or the most recent one. Always check whether its interval includes the queried date(s).\n"
)


def clamp_text(s: str, max_chars: int = MAX_TARGET_CHARS) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[:max_chars].rstrip()


def to_messages(example):
    """
    Converte un esempio mmTISER in formato chat per Qwen3-VL.
    Richiede che example abbia: question, image (HF Image path), output.
    """
    question = example.get("question", "")
    answer = clamp_text(example.get("output", "") or example.get("answer", ""))

    msg = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_MESSAGE},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            f"Question: {question}\n\n"
                            "Temporal context: The provided chart contains the temporal context "
                            "for this question.\nImportant: Use the chart to reason about the order, "
                            "overlap and duration of events, and answer exactly what is asked in the "
                            "question.\nWhen the question asks about a specific date or date range, "
                            "identify the event whose interval actually includes that date or fully "
                            "covers that range.\nIf the chart does not provide enough information to "
                            "answer, answer Unknown.\n"
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]
    }
    example["messages"] = msg["messages"]
    return example


def load_mm_tiser_df(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def train_eval_split(df: pd.DataFrame, test_size: float = 0.05, stratify_col: str = "datasetname", seed: int = 42):
    stratify = df[stratify_col] if stratify_col in df.columns else None
    train_df, eval_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify,
        random_state=seed,
    )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def to_hf_datasets(train_df: pd.DataFrame, eval_df: pd.DataFrame, image_column: str = "image"):
    hfd_train = Dataset.from_pandas(train_df)
    hfd_eval = Dataset.from_pandas(eval_df)

    hfd_train = hfd_train.shuffle(seed=42).map(to_messages)
    hfd_eval = hfd_eval.shuffle(seed=42).map(to_messages)

    hfd_train = hfd_train.cast_column(image_column, HFImage(decode=True))
    hfd_eval = hfd_eval.cast_column(image_column, HFImage(decode=True))

    return hfd_train, hfd_eval


def make_test_sample(df: pd.DataFrame, n_per_dataset: int = 60) -> pd.DataFrame:
    return df.groupby("datasetname").sample(n=n_per_dataset, random_state=42)
