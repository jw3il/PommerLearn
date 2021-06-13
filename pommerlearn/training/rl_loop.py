import subprocess
import threading
from multiprocessing import Process, Queue
from pathlib import Path
import time
from datetime import datetime
from typing import Optional

import concurrent.futures
import training.train_cnn
import copy
import shutil

from training.train_util import is_empty, rm_dir, move_content

# The path of the main executable for data generation
EXEC_PATH = Path("./PommerLearn")

# Every subdirectory will be put into the base dir
BASE_DIR = Path("./")

# The subdirectories for training and data generation
LOG_DIR = BASE_DIR / "log"
TRAIN_DIR = BASE_DIR / "train"
ARCHIVE_DIR = BASE_DIR / "archive"
MODEL_INIT_DIR = BASE_DIR / "model-init"
MODEL_OUT_DIR = BASE_DIR / "model-out"
MODEL_IN_DIR = BASE_DIR / "model"

WORKING_DIRS = [LOG_DIR, TRAIN_DIR, MODEL_OUT_DIR, MODEL_IN_DIR]

TENSORBOARD_DIR = BASE_DIR / "runs"

# Global variable used to stop the rl loop while it is running asynchronously
stop_rl = False
stop_rl_lock = threading.Lock()


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
        [f"./{str(EXEC_PATH)}", *local_args],
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


def train(data_dir: Path, out_dir: Path, torch_in_dir: Optional[str], train_config) -> concurrent.futures._base.Future:
    """
    Start a training pass.

    :param data_dir: The directory which contains the datatset
    :param out_dir: The output directory
    :param torch_in_dir: The torch input dir used to load an existing model
    :param train_config: The training config
    """

    # TODO: Add model input dir!?

    # fill the config
    local_train_config = copy.deepcopy(train_config)
    local_train_config["output_dir"] = str(out_dir)
    local_train_config["dataset_path"] = str(get_datatset(data_dir))
    local_train_config["torch_input_dir"] = torch_in_dir
    training.train_cnn.fill_default_config(local_train_config)

    # start training
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(lambda: training.train_cnn.train_cnn(local_train_config))

    return future


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


def rl_loop(run_id, max_iterations, dataset_args: list, train_config: dict, concurrency=False):
    """
    The main RL loop which alternates between data generation and training:

    generation 0 -> training 0 & generation 1 -> training 1 & generation 2 -> ...

    :param run_id: The (unique) id of the run, all data will be archived in ARCHIVE_DIR / run_id
    :param max_iterations: Max number of iterations (-1 for endless loop)
    :param dataset_args: Arguments for dataset generation
    :param train_config: Training configuration
    :param concurrency: Whether to generate and train concurrently
    (WARNING: This causes a delay of 1 iteration between sample generation and training)
    """

    global stop_rl

    # The current iteration
    it = 0

    def print_it(msg: str):
        print(f"{datetime.now()} > It. {it}: {msg}")

    # Before we can create a dataset, we need an initial model
    if is_empty(MODEL_INIT_DIR):
        training.train_cnn.export_initial_model(train_config, MODEL_IN_DIR)
        print("No initial model provided. Using new model.")
    else:
        shutil.copytree(MODEL_INIT_DIR, MODEL_IN_DIR)
        print("Using existing model.")

    last_model_dir_name = None

    # Loop: Train & Create -> Archive -> Train & Create -> ...
    train_future = None
    train_future_res = None
    # The first iteration does not count, as we only generate a dataset
    while max_iterations < 0 or it <= max_iterations:
        with stop_rl_lock:
            last_iteration = it == max_iterations or stop_rl

        if last_iteration:
            print_it("Entering the last iteration")

        # Start training if training data exists
        if not is_empty(TRAIN_DIR):
            print_it("Start training")
            train_future = train(TRAIN_DIR, MODEL_OUT_DIR, last_model_dir_name, train_config)
            # wait until the training is done before we start generating samples
            if not concurrency:
                train_future_res = train_future.result()
                print_it("Training done")
                move_content(MODEL_OUT_DIR, MODEL_IN_DIR)

        if not last_iteration:
            # Create a new dataset
            print_it("Create dataset")

            sproc_create_dataset = create_dataset(dataset_args)
            subprocess_verbose_wait(sproc_create_dataset)

            print_it("Dataset done")

        # Archive the model which was used to create the current data set with a smaller id
        archived_model_dir = ARCHIVE_DIR / run_id / (str(it - 1) + "_model")
        move_content(MODEL_IN_DIR, archived_model_dir)

        if concurrency and train_future is not None:
            train_future_res = train_future.result()
            print_it("Training done")
            move_content(MODEL_OUT_DIR, MODEL_IN_DIR)
            last_model_dir_name = str(MODEL_IN_DIR)
        else:
            last_model_dir_name = str(archived_model_dir)

        # Archive the old training dataset
        move_content(TRAIN_DIR, ARCHIVE_DIR / run_id)

        # Prepare the training dataset
        if not last_iteration:
            move_content(LOG_DIR, TRAIN_DIR)
            rename_datasets_id(TRAIN_DIR, str(it))
        else:
            if concurrency:
                # just directly archive the new model
                move_content(MODEL_IN_DIR, ARCHIVE_DIR / run_id / (str(it) + "_model"))

        it += 1

        if train_future_res is not None:
            # we executed a training step
            train_config["global_step"] = train_future_res["global_step"] + 1
            train_config["iteration"] += 1

    print("RL loop done")


def check_clean_working_dirs():
    """
    Ensures that all working directories are empty.
    """

    all_empty = all(is_empty(d) for d in WORKING_DIRS)
    if not all_empty:
        print("The working directories are not empty!")
        print("Before you continue, please inspect the directories and back up all valuable data.")
        print("After that, type 'clean' to clean up the working directories.")
        while True:
            cmd = input()
            if cmd == 'clean':
                print("Cleaning all working directories")
                for d in WORKING_DIRS:
                    rm_dir(d, keep_empty_dir=True)
                print("Done.")
                return
            else:
                print("Unknown command")


def clean_working_dirs():
    """
    Removes all empty working directories.
    """

    for d in WORKING_DIRS:
        if is_empty(d):
            rm_dir(d, keep_empty_dir=False)


def main():
    global stop_rl

    check_clean_working_dirs()

    # What is the purpose of the current run?
    run_comment = ""

    run_id = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if len(run_comment) > 0:
        run_id = run_id + f"-{run_comment}"

    # Info: All path-related arguments should be set inside the rl loop

    train_config = {
        "nb_epochs": 3,
        "test_size": 0.1,
        "tensorboard_dir": str(TENSORBOARD_DIR / run_id),
        "discount_factor": 0.99,
    }
    train_config = training.train_cnn.fill_default_config(train_config)

    model_type = "onnx"
    # model_type = "torch_cpu"
    # model_type = "torch_cuda"
    dataset_args = [
        "--mode=ffa_mcts",
        "--env_gen_seed_eps=10",
        "--max_games=-1",
        "--targeted_samples=50000",
        f"--model_dir={str(MODEL_IN_DIR / model_type)}"
    ]

    max_iterations = 5

    # Start the rl loop
    rl_thread = threading.Thread(target=rl_loop, args=(run_id, max_iterations, dataset_args, train_config))
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

    clean_working_dirs()


if __name__ == "__main__":
    main()
