from datasets import load_dataset

output_dir = "~/Data/conll2003/"


def main():
    dataset_name = "conll2003"
    train_data = load_dataset(dataset_name, split="train")
    test_data = load_dataset(dataset_name, split="test")
    dev_data = load_dataset(dataset_name, split="validation")

    train_data.to_json(output_dir + "train.json")
    test_data.to_json(output_dir + "test.json")
    dev_data.to_json(output_dir + "dev.json")


if __name__ == "__main__":
    main()
