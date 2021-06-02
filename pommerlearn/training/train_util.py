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