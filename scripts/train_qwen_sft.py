from data.dataset_tiser import load_mm_tiser_df, train_eval_split, to_hf_datasets
from modeling.sft_trainer import load_base_model, build_sft_trainer, run_training, save_and_push
from config import MM_TISER_TRAIN_JSON, QWEN_FINETUNED_ID


def main():
    df = load_mm_tiser_df(MM_TISER_TRAIN_JSON)
    train_df, eval_df = train_eval_split(df)
    train_ds, eval_ds = to_hf_datasets(train_df, eval_df)

    model, processor = load_base_model()
    trainer = build_sft_trainer(model, processor, train_ds, eval_ds)
    run_training(trainer)
    save_and_push(trainer, processor, QWEN_FINETUNED_ID)


if __name__ == "__main__":
    main()
