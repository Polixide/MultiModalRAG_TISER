# dataset paths
TISER_TRAIN_JSON = "TISER_train.json"
TISER_TEST_JSON = "TISER_test.json"
MM_TISER_TRAIN_JSON = "mm_TISER_train.json"
MM_TISER_TEST_JSON = "mm_TISER_test.json"
IMAGES_DIR = "mm_tiser_images"

# base / finetuned model ids
QWEN_BASE_ID = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_FINETUNED_ID = "Dancat/MM_Tiser_Qwen3_VL_FT_v2"

# SFT trainer hyperparams 
SFT_OUTPUT_DIR = "QWEN3_VL_8B_TISER_FT_v2"
SFT_NUM_EPOCHS = 1
SFT_PER_DEVICE_TRAIN_BATCH_SIZE = 2
SFT_GRADIENT_ACCUMULATION_STEPS = 4
SFT_GRADIENT_CHECKPOINTING = False
SFT_LEARNING_RATE = 1e-4
SFT_WARMUP_STEPS = 10
SFT_WEIGHT_DECAY = 0.01
SFT_MAX_GRAD_NORM = 1.0
SFT_BF16 = True
SFT_FP16 = False
SFT_TF32 = True
SFT_LR_SCHEDULER_TYPE = "cosine"
SFT_LOGGING_STEPS = 10
SFT_EVAL_STRATEGY = "steps"
SFT_EVAL_STEPS = 80
SFT_SAVE_STRATEGY = "steps"
SFT_SAVE_STEPS = 80
SFT_REMOVE_UNUSED_COLUMNS = False
SFT_RUN_NAME = "QWEN3_VL_8B_TISER_FT_v2"
SFT_TRACKIO_PROJECT = "QWEN3_VL_8B_TISER_FT_v2"
SFT_TRACKIO_SPACE = "MM_Tiser_Qwen3_VL_FT"

# collator / preprocess
MAX_TARGET_CHARS = 5000
MAX_IMAGE_SIDE = 1024
MAX_IMAGE_PIXELS = 1024 * 1024
MAX_LEN_TOKENS = 1024

# LoRA config 
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# RAG (Byaldi + ColQwen2) 
RAG_RETRIEVER_NAME = "vidore/colqwen2-v1.0"  
RAG_INDEX_NAME = "tiser_charts_index"
RAG_TOP_K = [1, 3, 5]
RAG_TOP1_DATASET_FILE = "mm_TISER_test_retrieval_top1.jsonl"
RAG_RECALL_OUTFILE = "mm_TISER_test_retrieval_top1.jsonl"
