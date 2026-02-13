from typing import List, Dict, Any

import torch
from PIL import Image

from config import (
    MAX_TARGET_CHARS,
    MAX_IMAGE_SIDE,
    MAX_IMAGE_PIXELS,
    MAX_LEN_TOKENS,
)


def clamp_text(s: str, max_chars: int = MAX_TARGET_CHARS) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[:max_chars].rstrip()


def resize_pil(pil: Image.Image, max_side: int = MAX_IMAGE_SIDE, max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    pil = pil.convert("RGB")
    w, h = pil.size

    scale_side = min(1.0, max_side / float(max(w, h)))
    scale_area = max_pixels / float(w * h) if (w * h) > max_pixels else 1.0
    scale = min(scale_side, scale_area)

    if scale >= 1.0:
        return pil

    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    pil = pil.resize((nw, nh), resample=Image.BICUBIC)
    return pil


def collate_fn(batch: List[Dict[str, Any]], processor, max_len: int = MAX_LEN_TOKENS) -> Dict[str, torch.Tensor]:
    # 1) testo completo (prompt + risposta)
    full_texts = [
        processor.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        for ex in batch
    ]

    # 2) solo prompt fino all'utente (per mascherare i token di input)
    prompt_texts = [
        processor.apply_chat_template(
            ex["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in batch
    ]

    # 3) immagini ridimensionate
    images = [resize_pil(ex["image"]) for ex in batch]

    # 4) tokenizzazione completa
    enc = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )

    input_ids = enc["input_ids"]
    pad_id = processor.tokenizer.pad_token_id

    # 5) lunghezze prompt (token-only)
    prompt_ids = processor.tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )["input_ids"]

    prompt_lens = (prompt_ids != pad_id).sum(dim=1)

    # 6) labels: copia di input_ids, ma maschera prompt e pad con -100
    labels = input_ids.clone()
    bs, seqlen = labels.shape

    for i in range(bs):
        pl = int(prompt_lens[i].item())
        pl = min(pl, seqlen)
        labels[i, :pl] = -100

    labels[labels == pad_id] = -100
    enc["labels"] = labels

    return enc
