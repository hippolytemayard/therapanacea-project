def read_txt_object(filepath: str) -> list:
    """Load txt object from filepath"""

    filepath = str(filepath)
    with open(filepath, "r") as output:
        return output.read().splitlines()
