from byaldi import RAGMultiModalModel

from rag.index_byaldi import load_rag_model
from rag.recall_metrics import compute_recall_at_k
from config import MM_TISER_TEST_JSON, RAG_RECALL_OUTFILE


def main():
    rag = load_rag_model()
    recall_global, recall_by_type = compute_recall_at_k(MM_TISER_TEST_JSON, rag, outfile=RAG_RECALL_OUTFILE)
    print("Global", recall_global)
    print("By_type", recall_by_type)


if __name__ == "__main__":
    main()
