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

# Global variable used to stop the rl loop
stop_rl = False
stop_rl_lock = threading.Lock()


def rm_dir(path: Path, keep_empty_dir=False):
    if not path.exists():
        return

    # recursively for every dir
    for child in path.iterdir():
        if child.is_dir():
            rm_dir(child)
        elif child.is_file():
            child.unlink()
        else:
            raise ValueError(f"Does not know how to remove {str(child)}!")

    # delete empty dir
    if not keep_empty_dir:
        path.rmdir()


def rename_subdirs_id(path: Path, id: str):
    if not path.exists() or not path.is_dir():
        return

    dir_count = 0

    # rename every subdir according to the given id
    for child in path.iterdir():
        if child.is_dir():
            child.rename(path / f"{id}_{dir_count}.zr")
            dir_count += 1


def move_content(source: Path, dest: Path):
    if not source.exists() or not source.is_dir():
        return

    dest.mkdir(exist_ok=True, parents=True)

    for child in source.iterdir():
        child.replace(dest / child.name)


def is_empty(dir: Path):
    if not dir.exists():
        return True

    if not dir.is_dir():
        raise ValueError(f"{str(dir)} is no directory!")

    for _ in dir.iterdir():
        return False

    return True


def create_dataset(arguments):
    # execute the c++ program once

    # clear the dir if it already exists
    rm_dir(LOG_DIR, keep_empty_dir=True)
    # make sure it exists
    LOG_DIR.mkdir(exist_ok=True)

    local_args = copy.deepcopy(arguments)
    local_args.append([
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
    if not dir.exists() or not dir.is_dir():
        raise ValueError(f"{str(dir)} is no directory!")

    for child in dir.iterdir():
        if child.is_dir() and child.name.endswith(".zr"):
            return child

    raise ValueError(f"Could not find any dataset in {dir}!")


def train(data_dir, model_dir, train_config):
    # fill the config
    local_train_config = copy.deepcopy(train_config)
    local_train_config["model_output_dir"] = str(model_dir)
    local_train_config["dataset_path"] = str(get_datatset(data_dir))
    training.train_cnn.fill_default_config(local_train_config)

    # start training
    train_thread = threading.Thread(target=training.train_cnn.train_cnn(local_train_config), args=(train_config,))
    train_thread.start()

    return train_thread


def create_initial_models(model_dir):
    # TODO: Use train_cnn and build initial model if the model dir is empty!
    Path.mkdir(model_dir / "example-model", parents=True, exist_ok=True)


def rl_loop(max_iterations, dataset_args, train_config):
    global stop_rl
    # The current iteration
    it = 0

    def print_it(msg: str):
        print(f"{datetime.now()} > It. {it}: {msg}")

    def get_identifier() -> str:
        return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # Before we can create any datasets, we have to create initial models
    create_initial_models(MODEL_IN_DIR)
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
            sproc_create_dataset.wait()

            print_it("Dataset done")

        # Archive previous models
        move_content(MODEL_IN_DIR, ARCHIVE_DIR / ("model_" + iteration_id))

        # Wait until the training is done as well
        if thread_train is not None:
            thread_train.join()
            print_it("Training done")
            MODEL_OUT_DIR.rename(MODEL_IN_DIR)

        # Archive the old training dataset
        iteration_id = get_identifier()
        move_content(TRAIN_DIR, ARCHIVE_DIR)

        # Prepare the training dataset
        if not last_iteration:
            move_content(LOG_DIR, TRAIN_DIR)
            rename_subdirs_id(TRAIN_DIR, iteration_id)
        else:
            # just directly archive the created model
            move_content(MODEL_IN_DIR, ARCHIVE_DIR / ("model_" + iteration_id))

        it += 1

    print_it("RL loop done")


def main():
    global stop_rl

    # Info: All path-related arguments should be set inside the rl loop

    dataset_args = [
        "--mode=ffa_sl",
        "--max_games=10",
    ]

    train_config = {
        "nb_epochs": 1
    }

    max_iterations = 3

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
