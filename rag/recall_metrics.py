import json
from collections import defaultdict

from tqdm.auto import tqdm

from config import RAG_TOP_K, RAG_RECALL_OUTFILE


def compute_recall_at_k(dataset_file: str, rag, outfile: str = RAG_RECALL_OUTFILE, ks=None):
    if ks is None:
        ks = RAG_TOP_K

    stats = {
        "total": 0.0,
        "perk": {k: 0.0 for k in ks},
    }
    per_type = {
        "gantt": {"total": 0.0, "perk": {k: 0.0 for k in ks}},
        "scatter": {"total": 0.0, "perk": {k: 0.0 for k in ks}},
        "line": {"total": 0.0, "perk": {k: 0.0 for k in ks}},
    }

    id2file = rag.get_docids_to_filenames()

    outlines = []
    maxk = max(ks)

    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Eval RAG"):
            data = json.loads(line)
            query = data.get("question")
            image_path = data.get("image")
            chart_type = data.get("chart_type", None)
            answer = data.get("answer")

            if not image_path:
                continue

            retrieved = rag.search(query, k=maxk)

            stats["total"] += 1.0
            if chart_type in stats["per_type"]:
                stats["per_type"][chart_type]["total"] += 1.0

            retrieved_files = []
            for r in retrieved:
                docid = r["doc_id"]
                retrieved_files.append(id2file[docid])

            top1_path = retrieved_files[0]
            outlines.append(
                {
                    "question": query,
                    "gold_image": image_path,
                    "gold_answer": answer,
                    "chart_type": chart_type,
                    "top1_path": top1_path,
                }
            )

            for k in ks:
                topk_files = set(retrieved_files[:k])
                hit = image_path in topk_files
                if hit:
                    stats["per_k"][k] += 1.0
                    if chart_type in stats["per_type"]:
                        stats["per_type"][chart_type]["per_k"][k] += 1.0

    recall_global = {k: (stats["per_k"][k] / stats["total"] if stats["total"] > 0 else 0.0) for k in ks}
    recall_by_type = {}
    for t, d in per_type.items():
        recall_by_type[t] = {
            k: (d["per_k"][k] / d["total"] if d["total"] > 0 else 0.0) for k in ks
        }

    with open(outfile, "w", encoding="utf-8") as fout:
        for row in outlines:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    return recall_global, recall_by_type
