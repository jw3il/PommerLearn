import re
from pathlib import Path


def rm_dir(dir: Path, keep_empty_dir=True):
    """
    Removes all the content from the given directory (and optionally also the directory itself)

    :param dir: A directory
    :param keep_empty_dir: Whether to keep the empty directory after all its content has been deleted
    """
    if not dir.exists() or not dir.is_dir():
        return

    # recursively for every dir
    for child in dir.iterdir():
        if child.is_dir():
            rm_dir(child, keep_empty_dir=False)
        elif child.is_file():
            child.unlink()
        else:
            raise ValueError(f"Does not know how to remove {str(child)}!")

    # delete empty dir
    if not keep_empty_dir:
        dir.rmdir()


def rm_files_with_type(dir: Path, file_suffix: str):
    """
    Removes all files from the given directory that end with the given suffix.

    :param dir: A directory
    :param file_suffix: A suffix, e.g. ".txt"
    """
    if not dir.exists() or not dir.is_dir():
        return

    # recursively for every child
    for child in dir.iterdir():
        if child.is_dir():
            rm_files_with_type(child, file_suffix)
        elif child.is_file() and child.suffix == file_suffix:
            child.unlink()


def move_content(source: Path, dest: Path):
    """
    Move all content from the source to the destination directory.

    :param source: The source directory
    :param dest: The destination directory
    """
    if not source.exists() or not source.is_dir():
        return

    dest.mkdir(exist_ok=True, parents=True)

    for child in source.iterdir():
        child.replace(dest / child.name)


def is_empty(dir: Path):
    """
    Check whether the specified directory is empty (or does not exists).

    :param dir: A directory
    :return: dir.exists() and dir is empty
    """
    if not dir.exists():
        return True

    if not dir.is_dir():
        raise ValueError(f"{str(dir)} is no directory!")

    for _ in dir.iterdir():
        return False

    return True


# idea from https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def natural_keys(text: str):
    """
    Split the given string into keys that allow for human/natural sorting.
    This means that all numbers are sorted based on their value, not on their textual representation.

    Example:
        the strings "hello_20_2", "hello_3_2", "hello_0_0" are mapped to the following keys
        ["hello", "_", 20, "_", 2], ["hello", "_", 3, "_", 2], ["hello", "_", 0, "_", 0]
        that allow for correct sorting: hello_20_2 > hello_3_2 > hello_0_0

    :param text: The string that should be converted
    :returns: The keys for text
    """
    def try_convert_int(t: str):
        return int(t) if t.isdigit() else t

    return [try_convert_int(t) for t in re.split(r'(\d+)', text)]
