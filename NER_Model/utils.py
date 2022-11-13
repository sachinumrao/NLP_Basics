import jsonlines


def load_jsonl_data(file_path):
    """
    Function to read jsonl format data.
    """
    data = []
    with jsonlines.open(file_path, "r") as reader:
        for line in reader:
            data.append(line)

    return data
