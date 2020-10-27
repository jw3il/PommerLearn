import subprocess
import threading
from multiprocessing import Process, Queue
from pathlib import Path
import time
from datetime import datetime
import training.train_cnn
import copy

EXEC_NAME = Path("./PommerLearn")

LOG_DIR = Path("./log/")
TRAIN_DIR = Path("./train/")
ARCHIVE_DIR = Path("./archive/")
MODEL_INIT_DIR = Path("./model-init/")
MODEL_OUT_DIR = Path("./model-out/")
MODEL_IN_DIR = Path("./model/")

# Global variable used to stop the rl loop while it is running asynchronously
stop_rl = False
stop_rl_lock = threading.Lock()


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
            rm_dir(child)
        elif child.is_file():
            child.unlink()
        else:
            raise ValueError(f"Does not know how to remove {str(child)}!")

    # delete empty dir
    if not keep_empty_dir:
        dir.rmdir()


def rename_datasets_id(dir: Path, id: str):
    """
    Rename all the datasets inside a directory according to the given id.
    
    :param dir: A directory which might contain some datatsets
    :param id: An identifier for these datasets
    """
    if not dir.exists() or not dir.is_dir():
        return

    dir_count = 0

    # rename every subdir according to the given id
    for child in dir.iterdir():
        if child.is_dir() and child.name.endswith(".zr"):
            child.rename(dir / f"{id}_{dir_count}.zr")
            dir_count += 1


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


def create_dataset(arguments):
    """
    Create a dataset by executing the C++ program.

    :param arguments: The program arguments (excluding log and file dirs)
    """
    # clear the log dir if it already exists
    rm_dir(LOG_DIR, keep_empty_dir=True)
    # make sure it exists
    LOG_DIR.mkdir(exist_ok=True)

    local_args = copy.deepcopy(arguments)
    local_args.extend([
        "--log",
        f"--file_prefix={str(LOG_DIR / Path('data'))}"
    ])

    proc = subprocess.Popen(
        [f"./{EXEC_NAME}", *local_args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False
    )

    return proc


def get_datatset(dir: Path) -> Path:
    """
    Get the path of the dataset inside a directory.

    :param dir: Some directory which contains a dataset
    :return: the first subdirectory within dir ending with .zr
    """
    if not dir.exists() or not dir.is_dir():
        raise ValueError(f"{str(dir)} is no directory!")

    for child in dir.iterdir():
        if child.is_dir() and child.name.endswith(".zr"):
            return child

    raise ValueError(f"Could not find any dataset in {dir}!")


def train(data_dir, out_dir, train_config):
    """
    Start a training pass.

    :param data_dir: The directory which contains the datatset
    :param out_dir: The output directory
    :param train_config: The training config
    """

    # TODO: Add model input dir!?

    # fill the config
    local_train_config = copy.deepcopy(train_config)
    local_train_config["output_dir"] = str(out_dir)
    local_train_config["dataset_path"] = str(get_datatset(data_dir))
    training.train_cnn.fill_default_config(local_train_config)

    # start training
    train_thread = threading.Thread(target=training.train_cnn.train_cnn(local_train_config), args=(train_config,))
    train_thread.start()

    return train_thread


def create_initial_models(model_dir: Path, batch_sizes):
    """
    Create initial models if the given directory is not empty.

    :param model_dir: Directory where the models should be created
    :param batch_sizes: The batch sizes
    """
    if is_empty(model_dir):
        training.train_cnn.export_initial_model(batch_sizes, model_dir)
    else:
        print("Skipping initialization because model dir is not empty!")


def subprocess_verbose_wait(sproc):
    while sproc.poll() is None:
        try:
            sproc.wait(1)
        except subprocess.TimeoutExpired:
            pass

        while True:
            line = sproc.stdout.readline()
            if not line:
                break
            print(line.decode("utf-8"), end='')


def rl_loop(max_iterations, dataset_args: list, train_config: dict):
    """
    The main RL loop which alternates between data generation and training:

    generation 0 -> training 0 & generation 1 -> training 1 & generation 2 -> ...

    :param max_iterations: Max number of iterations (-1 for endless loop)
    :param dataset_args: Arguments for dataset generation
    :param train_config: Training configuration
    """

    global stop_rl

    # The current iteration
    it = 0

    def print_it(msg: str):
        print(f"{datetime.now()} > It. {it}: {msg}")

    def get_identifier() -> str:
        return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # Before we can create any datasets, we have to create initial models
    create_initial_models(MODEL_IN_DIR, train_config["model_batch_sizes"])
    iteration_id = "init_" + get_identifier()

    # Loop: Train & Create -> Archive -> Train & Create -> ...
    thread_train = None
    while max_iterations < 0 or it < max_iterations:
        # TODO: Clean exit instead of break
        with stop_rl_lock:
            if stop_rl:
                break

        last_iteration = it == max_iterations - 1
        if last_iteration:
            print_it("Entering the last iteration")

        # Start training if training data exists
        if not is_empty(TRAIN_DIR):
            print_it("Start training")
            thread_train = train(TRAIN_DIR, MODEL_OUT_DIR, train_config)

        if not last_iteration:
            # Create a new dataset
            print_it("Create dataset")

            sproc_create_dataset = create_dataset(dataset_args)
            subprocess_verbose_wait(sproc_create_dataset)

            print_it("Dataset done")

        # Archive previous models
        move_content(MODEL_IN_DIR, ARCHIVE_DIR / ("model_" + iteration_id))

        # Wait until the training is done as well
        if thread_train is not None:
            thread_train.join()
            print_it("Training done")
            move_content(MODEL_OUT_DIR, MODEL_IN_DIR)

        # Archive the old training dataset
        iteration_id = get_identifier()
        move_content(TRAIN_DIR, ARCHIVE_DIR)

        # Prepare the training dataset
        if not last_iteration:
            move_content(LOG_DIR, TRAIN_DIR)
            rename_datasets_id(TRAIN_DIR, iteration_id)
        else:
            # just directly archive the created model
            move_content(MODEL_IN_DIR, ARCHIVE_DIR / ("model_" + iteration_id))

        it += 1

    print_it("RL loop done")


def main():
    global stop_rl

    # Info: All path-related arguments should be set inside the rl loop

    train_config = {
        "nb_epochs": 1
    }
    training.train_cnn.fill_default_config(train_config)

    use_cuda_models = True
    dataset_args = [
        "--mode=ffa_mcts",
        "--max_games=1",
        f"--model_dir={training.train_cnn.get_model_path(MODEL_IN_DIR, use_cuda_models)}"
    ]

    max_iterations = 5

    # Start the rl loop
    rl_thread = threading.Thread(target=rl_loop, args=(max_iterations, dataset_args, train_config))
    rl_thread.start()

    # Allow early stopping
    while rl_thread.is_alive():
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            with stop_rl_lock:
                stop_rl = True

            print("Stopping at the end of this iteration.")
            break

    rl_thread.join()

    # Clean up the empty dirs
    def clean(dir: Path):
        if is_empty(dir):
            rm_dir(dir)

    clean(LOG_DIR)
    clean(TRAIN_DIR)
    clean(MODEL_IN_DIR)
    clean(MODEL_OUT_DIR)


if __name__ == "__main__":
    main()
