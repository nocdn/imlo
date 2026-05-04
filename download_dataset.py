import logging

from coursework_data import download_pet_dataset


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Downloading Oxford-IIIT Pet dataset official splits...")
    trainval_dataset, test_dataset = download_pet_dataset()

    logging.info("Download complete.")
    logging.info("Trainval images: %d", len(trainval_dataset))
    logging.info("Test images: %d", len(test_dataset))
    logging.info("Classes: %d", len(trainval_dataset.classes))
    logging.info("Data directory: %s", trainval_dataset.root)


if __name__ == "__main__":
    main()
