from pathlib import Path


# inspired by https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
def check_dir(string: str):
    """
    Checks if the given string is a directory. Raises an Error if that's not the case.

    :param string: A string
    """
    if Path(string).is_dir():
        return string
    else:
        raise NotADirectoryError(string)


def check_file(string: str):
    """
    Checks if the given string is a file. Raises an Error if that's not the case.

    :param string: A string
    """
    if Path(string).is_file():
        return string
    else:
        raise FileNotFoundError(string)
