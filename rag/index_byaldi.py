import os

from byaldi import RAGMultiModalModel

from config import RAG_RETRIEVER_NAME, RAG_INDEX_NAME, IMAGES_DIR


def load_rag_model(model_name: str = RAG_RETRIEVER_NAME) -> RAGMultiModalModel:
    rag = RAGMultiModalModel.from_pretrained(model_name)
    return rag


def build_index(image_folder: str = IMAGES_DIR, index_name: str = RAG_INDEX_NAME, overwrite: bool = True):
    rag = load_rag_model()
    rag.index(
        input_path=image_folder,
        index_name=index_name,
        store_collection_with_index=True,
        overwrite=overwrite,
    )
    return rag
