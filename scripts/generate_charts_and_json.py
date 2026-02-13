from data.charts_generator import generate_dataset_parallel
from config import (
    TISER_TRAIN_JSON,
    TISER_TEST_JSON,
    IMAGES_DIR,
    MM_TISER_TRAIN_JSON,
    MM_TISER_TEST_JSON,
)


def main():
    if TISER_TRAIN_JSON:
        generate_dataset_parallel(TISER_TRAIN_JSON, IMAGES_DIR, MM_TISER_TRAIN_JSON)
    if TISER_TEST_JSON:
        generate_dataset_parallel(TISER_TEST_JSON, IMAGES_DIR, MM_TISER_TEST_JSON)


if __name__ == "__main__":
    main()
