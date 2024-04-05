def read_txt_object(filepath: str) -> list:
    """Load txt object from filepath"""

    filepath = str(filepath)
    with open(filepath, "r") as output:
        return output.read().splitlines()


def write_predictions_to_file(predictions: list[int], file_path: str):
    """Write prediction to txt file"""
    with open(file_path, "w") as file:
        for prediction in predictions:
            file.write(f"{prediction}\n")
