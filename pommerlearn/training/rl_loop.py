import argparse
import subprocess
import sys
import os
import logging
import threading
from pathlib import Path
import time
from datetime import datetime
from typing import Optional, Iterator, List, Iterable
import concurrent.futures

import numpy as np
import numpy.random

import copy
import shutil
from rtpt.rtpt import RTPT

import training.train_cnn
from training.train_util import is_empty, rm_dir, move_content, natural_keys
from training.util_argparse import check_dir, check_file

# Global variable used to stop the rl loop while it is running asynchronously
stop_rl = False
stop_rl_lock = threading.Lock()


def change_binary_name(binary_dir: str, current_binary_name: str, process_name: str, nn_update_idx: int,
                       overwrite: bool):
    """
    Change the name of the binary to the process' name (which includes initials,
    binary name and remaining time) & additionally add the current epoch.
    (based on implementation by maxalexger (GitHub))
    :return: the new binary name
    """
    idx = process_name.find(f'#')
    new_binary_name = f'{process_name[:idx]}_UP={nn_update_idx}{process_name[idx:]}'

    if os.path.exists(binary_dir + new_binary_name) and new_binary_name != current_binary_name:
        if overwrite:
            logging.info(f'Binary with target name {new_binary_name} already exists. Removing old binary.')
            os.remove(binary_dir + new_binary_name)
        else:
            raise ValueError(f"Binary with target name {new_binary_name} already exists.")

    if new_binary_name != current_binary_name:
        os.rename(binary_dir + current_binary_name, binary_dir + new_binary_name)
        logging.info(f'Changed binary name to: {new_binary_name}')

    return new_binary_name


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


def create_dataset(exec_path, log_dir, arguments, model_dir: Path, model_subdir: str):
    """
    Create a dataset by executing the C++ program.

    :param exec_path: The path to the executable that generates training data
    :param log_dir: Where to place the generated dataset
    :param arguments: The program arguments (excluding log and file dirs)
    :param model_dir: The main directory of the model
    :param model_subdir: The relevant subdirectory inside model_dir
    """
    # clear the log dir if it already exists
    rm_dir(log_dir, keep_empty_dir=True)
    # make sure it exists
    log_dir.mkdir(exist_ok=True)

    local_args = copy.deepcopy(arguments)
    local_args.extend([
        "--log",
        f"--file_prefix={str(log_dir / Path('data'))}",
        f"--model_dir={str(model_dir / model_subdir)}",
    ])

    print("Args: ", " ".join(local_args))
    proc = subprocess.Popen(
        [f"{str(exec_path.absolute())}", *local_args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return proc


def sort_paths(paths: Iterable[Path]) -> List[Path]:
    """
    Sorts the given paths.

    :param paths: A list of paths
    :returns: the sorted path list
    """
    return sorted(paths, key=lambda x: natural_keys(str(x)))


def get_datatsets(dir: Path) -> List[Path]:
    """
    Get all datasets in a given directory.

    :param dir: A directory which contains datasets
    :return: all subdirectories inside dir ending with ".zr" (sorted)
    """
    if not dir.exists() or not dir.is_dir():
        raise ValueError(f"{str(dir)} is no directory!")

    return list(filter(lambda p: p.is_dir and p.name.endswith(".zr"), [p for p in dir.iterdir()]))


def train(sorted_dataset_paths: List[Path], out_dir: Path, torch_in_dir: Optional[str], train_config
          ) -> concurrent.futures._base.Future:
    """
    Start a training pass.

    :param sorted_dataset_paths: Dataset paths sorted by age (descending)
    :param archive_data_dir: Directory with old training data
    :param out_dir: The output directory
    :param torch_in_dir: The torch input dir used to load an existing model
    :param train_config: The training config
    """

    # fill the config
    local_train_config = copy.deepcopy(train_config)
    local_train_config["output_dir"] = str(out_dir)

    # use last x datasets
    # important: only the last dataset will be logged to tensorboard
    local_train_config["dataset_path"] = [str(path) for path in sorted_dataset_paths]

    local_train_config["torch_input_dir"] = torch_in_dir
    training.train_cnn.fill_default_config(local_train_config)

    # start training
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(lambda: training.train_cnn.train_cnn(local_train_config))

    return future


def subprocess_verbose_wait(sproc):
    def print_stream(stream, file):
        while True:
            line = stream.readline()
            if not line:
                break
            print(line.decode("utf-8"), end='', file=file)

    return_code = sproc.poll()
    while return_code is None:
        try:
            sproc.wait(1)
        except subprocess.TimeoutExpired:
            pass

        print_stream(sproc.stdout, sys.stdout)
        print_stream(sproc.stderr, sys.stderr)

        return_code = sproc.poll()

    # TODO: include when random cuda errors at driver shutdown are fixed..
    # if return_code != 0:
    #     raise RuntimeError(f"Subprocess returned {return_code}!")


def choose_tail_and_random_from_end(li: List, num_tail: int, num_rand: int, rand_include: float):
    """
    Chooses the last num_tail elements in the given list and randomly num_rand elements without replacement preferably
    from the last rand_include * (len(li) - num_tail) elements in the list. Selects further elements from the tail if
    that's not possible.

    Examples:
        choose_tail_and_random_from_end([1, 2], 3, 3, 0.1) will return [1, 2] because there are only 2 elements in
        the list.

        choose_tail_and_random_from_end([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, 3, 0.1) may return [7, 6, 5, 8, 9, 10].
        It first selects the last three elements [8, 9, 10]. The last 10% of the remaining list [1, 2, 3, 4, 5, 6, 7]
        (rounded up) just consist of the element 7. Therefore, we randomly add additional elements from the tail without
        replacement until we reach num_rand=3 elements.

        choose_tail_and_random_from_end([1, ..., 100], 3, 3, 0.1) may return [92, 95, 93, 97, 98, 99].

    :param li: A list
    :param num_tail: The number of elements to select starting at the end of the list
    :param num_rand: The number of elements that are chosen randomly from the last elements
    :param rand_include: How much of the list (starting at the back) should be included in the random selection
        (0.2 means that samples will be chosen from the last 20% of the list).
    :returns: the filtered list
    """
    max_index_exclusive = len(li) - num_tail
    min_index_inclusive = int(np.floor(max_index_exclusive * (1 - rand_include)))
    diff = max_index_exclusive - min_index_inclusive

    if diff < num_rand:
        # fill up elements outside rand_include until we have num_rand elements
        min_index_inclusive = max(0, min_index_inclusive - (num_rand - diff))
        diff = max_index_exclusive - min_index_inclusive

    if diff > 0 and num_rand > 0:
        indices = min_index_inclusive + numpy.random.choice(diff, min(num_rand, diff), replace=False)
        return [li[i] for i in indices] + li[-num_tail:]

    return li[-num_tail:]


def rl_loop(run_id, max_iterations, base_dir: Path, exec_path: Path, dataset_args: list, train_config: dict,
            model_subdir: str, num_datasets_latest: int, num_datasets_recent: int, datasets_recent_include: float,
            rtpt: RTPT):
    """
    The main RL loop which alternates between data generation and training:

    generation 0 -> training 0 & generation 1 -> training 1 & generation 2 -> ...

    :param run_id: The (unique) id of the run, all data will be archived in ARCHIVE_DIR / run_id
    :param max_iterations: Max number of iterations (-1 for endless loop)
    :param base_dir: The base directory where all generated files will be placed
    :param exec_path: The path to the executable that generates training data
    :param dataset_args: Arguments for dataset generation
    :param train_config: Training configuration
    :param model_subdir: The name of the subdirectory inside the model dir used for sample generation
    :param num_datasets_latest: The number of last datasets that are used for training in each iteration
    :param num_datasets_recent: The number of "recent" datasets that will be used
    :param datasets_recent_include: Defines what "recent" means, proportion of the collected datasets (e.g. 0.1 means
        that we prefer to select datasets from the most recent 10% of all datasets).
    (WARNING: This causes a delay of 1 iteration between sample generation and training)
    :param rtpt: RTPT object
    """

    global stop_rl

    archive_dir = base_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    log_dir = base_dir / "log"
    log_dir.mkdir(exist_ok=True)

    model_init_dir = base_dir / "model-init"

    # copy the executable and rename this copy according to rtpt (to be removed after the rl loop)
    # => if this process is killed, we don't have to manually rename the original executable again
    exec_copy_path = exec_path.parent.absolute() / (exec_path.stem + "WorkingCopy")
    if exec_copy_path.exists():
        exec_copy_path.unlink()
    shutil.copy2(exec_path, exec_copy_path)

    # The current iteration
    it = 0

    def print_it(msg: str):
        print(f"{datetime.now()} > It. {it}: {msg}")

    run_archive_dir = archive_dir / run_id
    run_archive_dir.mkdir(exist_ok=True)

    last_model_dir = None
    model_dir = run_archive_dir / (str(it - 1) + "_model")

    # Before we can create a dataset, we need an initial model
    if is_empty(model_init_dir):
        model_dir.mkdir(exist_ok=True)
        training.train_cnn.export_initial_model(train_config, model_dir)
        print("No initial model provided. Using new model.")
    else:
        shutil.copytree(str(model_init_dir), model_dir)
        print("Using existing model.")

    # Loop: Train & Create -> Archive -> Train & Create -> ...
    train_future_res = None
    # The first iteration does not count, as we only generate a dataset
    while max_iterations < 0 or it <= max_iterations:
        model_dir = run_archive_dir / (str(it - 1) + "_model")

        with stop_rl_lock:
            last_iteration = it == max_iterations or stop_rl

        if last_iteration:
            print_it("Entering the last iteration")

        datasets = get_datatsets(run_archive_dir)

        # Start training if training data exists
        if len(datasets) > 0:
            print_it("Start training")

            # filter datasets
            datasets = sort_paths(datasets)
            datasets = choose_tail_and_random_from_end(datasets, num_datasets_latest, num_datasets_recent,
                                                       datasets_recent_include)
            datasets = sort_paths(datasets)

            train_future = train(datasets, model_dir, last_model_dir, train_config)
            # wait until the training is done before we start generating samples
            train_future_res = train_future.result()
            print_it("Training done")
            
            # Update the RTPT (subtitle is optional)
            rtpt.step(subtitle=f"global_step={train_config['global_step']:d}")

        # Create a new dataset using the current model
        if not last_iteration:
            # Create a new dataset
            print_it("Create dataset")

            exec_copy_path = adjust_exec_path(exec_copy_path, rtpt._get_title(), train_config["iteration"])
            sproc_create_dataset = create_dataset(exec_copy_path, log_dir, dataset_args, model_dir, model_subdir)
            subprocess_verbose_wait(sproc_create_dataset)

            print_it("Dataset done")

            # rename the dataset according to the iteration and move it into the archive
            rename_datasets_id(log_dir, str(it))
            move_content(log_dir, run_archive_dir)

        last_model_dir = model_dir
        it += 1

        if last_iteration:
            print("Reached end of last iteration")
            break

        if train_future_res is not None:
            # we executed a training step
            train_config["global_step"] = train_future_res["global_step"] + 1
            train_config["iteration"] += 1

    print("RL loop done")
    exec_copy_path.unlink()


def adjust_exec_path(exec_path: Path, rtpt_title: str, iteration: int) -> Path:
    """
    Changes the binary name and returns an adjusted version of the exec path based on iteration number and rtpt title.
    :param exec_path: Execution path / binary file path
    :param rtpt_title: Title of the rtpt object
    :param iteration: Iteration index
    :return: Adjusted rtpt title
    """
    current_binary_name = str(exec_path).split(os.sep)[-1]
    binary_dir = str(exec_path)[:-len(current_binary_name)]
    new_binary_name = change_binary_name(binary_dir, current_binary_name, rtpt_title, iteration, True)
    exec_path = Path(binary_dir + new_binary_name)
    return exec_path


def check_clean_working_dirs(working_dirs: List[Path]):
    """
    Ensures that all working directories are empty.

    :param working_dirs: List of working directories
    """

    all_empty = all(is_empty(d) for d in working_dirs)
    if not all_empty:
        print("The working directories are not empty!")
        print("Before you continue, please inspect the directories and back up all valuable data.")
        print("After that, type 'clean' to clean up the working directories.")
        while True:
            cmd = input()
            if cmd == 'clean':
                print("Cleaning all working directories")
                for d in working_dirs:
                    rm_dir(d, keep_empty_dir=True)
                print("Done.")
                return
            else:
                print("Unknown command")


def clean_working_dirs(working_dirs: List[Path]):
    """
    Removes all empty working directories.

    :param working_dirs: List of working directories
    """

    for d in working_dirs:
        if is_empty(d):
            rm_dir(d, keep_empty_dir=False)


def get_working_dirs(base_dir: Path):
    """
    Get the working directories for the given base directory.

    :param base_dir: The base directory
    """
    return [base_dir / "log"]


def main():
    global stop_rl

    parser = argparse.ArgumentParser(description='PommerLearn RL Loop')
    parser.add_argument('--dir', default='.', type=check_dir,
                        help='The main training directory that is used to store all intermediate and archived results')
    parser.add_argument('--exec', default='./PommerLearn', type=check_file,
                        help='The path to the PommerLearn executable')
    parser.add_argument('--it', default=100, type=int,
                        help='Maximum number of iterations (-1 for endless run that has to be stopped manually)')
    parser.add_argument('--num-latest', default=4, type=int,
                        help='The number of last datasets that are used for training in each iteration')
    parser.add_argument('--num-recent', default=4, type=int,
                        help='The number of "recent" datasets that will be used in addition to --num-latest')
    parser.add_argument('--recent-include', default=0.1, type=float,
                        help='Defines the meaning of "recent" as a proportion of all datasets (e.g. 0.1 for last 10%)')
    parser.add_argument('--name-initials', default='XX', type=str,
                        help='The name initials that are used to specify the user for the RTPT library.')

    parsed_args = parser.parse_args()

    base_dir = Path(parsed_args.dir)
    exec_path = Path(parsed_args.exec)

    working_dirs = get_working_dirs(base_dir)

    check_clean_working_dirs(working_dirs)

    # What is the purpose of the current run?
    run_comment = ""

    run_id = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if len(run_comment) > 0:
        run_id = run_id + f"-{run_comment}"

    # Info: All path-related arguments should be set inside the rl loop

    value_version = 1
    train_config = {
        "nb_epochs": 2,
        "only_test_last": True,
        "test_size": 0.5,
        "tensorboard_dir": str(base_dir / "runs" / run_id),
        "discount_factor": 0.97,
        "mcts_val_weight": 0.3,
        "value_version": value_version,
        # "train_sampling_mode": "weighted_value_class",
        # for lstm
        "use_flat_core": False,
        "use_lstm": False,
        "sequence_length": 8,
    }
    train_config = training.train_cnn.fill_default_config(train_config)

    model_subdir = "onnx"
    # model_subdir = "torch_cpu"
    # model_subdir = "torch_cuda"
    dataset_args = [
        "--mode=ffa_mcts",
        "--env_gen_seed_eps=2",
        "--max_games=-1",
        "--targeted_samples=50000",
        "--state_size=0",
        "--planning_agents=SimpleUnbiasedAgent",  # LazyAgent
        "--simulations=100",
        "--movetime=100",
        f"--value_version={value_version}",
    ]

    # Create RTPT object
    rtpt = RTPT(name_initials=parsed_args.name_initials, experiment_name='Pommer', max_iterations=parsed_args.it)

    # Start the RTPT tracking
    rtpt.start()

    # Start the rl loop
    rl_args = (run_id, parsed_args.it, base_dir, exec_path, dataset_args, train_config, model_subdir,
               parsed_args.num_latest, parsed_args.num_recent, parsed_args.recent_include, rtpt)
    rl_thread = threading.Thread(target=rl_loop, args=rl_args)
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

    clean_working_dirs(working_dirs)


if __name__ == "__main__":
    main()
