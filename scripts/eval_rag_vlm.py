import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from config import QWEN_FINETUNED_ID, RAG_TOP1_DATASET_FILE
from rag.rag_vlm_eval import eval_rag_plus_vlm


def main():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_FINETUNED_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(QWEN_FINETUNED_ID)
    metrics = eval_rag_plus_vlm(RAG_TOP1_DATASET_FILE, model, processor)
    print(metrics)


if __name__ == "__main__":
    main()
