import argparse
import pprint
import re
import subprocess
import sys
import os
import logging
import threading
from math import floor
from pathlib import Path
import time
from datetime import datetime
from typing import Optional, Iterator, List, Iterable
import concurrent.futures

import numpy as np
import numpy.random

import copy
import shutil

import torch.cuda
from rtpt.rtpt import RTPT

import training.train_cnn
from training.train_util import is_empty, rm_dir, move_content, natural_keys, rm_files_with_type
from training.util_argparse import check_dir, check_file
import warnings

# Global variable used to stop the rl loop while it is running asynchronously
stop_rl = False
stop_rl_lock = threading.Lock()


def rename_file(file_path: Path, new_filename: str, overwrite: bool):
    """
    Renames a file with explicit overwrite check.

    :param file_path: Complete path to the file
    :param new_filename: New filename (only the name without the path)
    :param overwrite: Whether to overwrite existing files.
    :return: The path to the new file
    """
    destination_path = file_path.parent / new_filename
    if destination_path.exists() and destination_path.name != file_path.name:
        if overwrite:
            logging.info(f'File with target name {new_filename} already exists. Removing old file.')
            destination_path.unlink()
        else:
            raise ValueError(f"File with target name {new_filename} already exists.")

    if destination_path.name != file_path.name:
        file_path.rename(destination_path)
        new_path = destination_path
        logging.info(f'Changed binary name to: {new_filename}')
    else:
        new_path = file_path

    return new_path


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


def create_dataset(exec_path, file_prefix: str, arguments, model_dir: Path, model_subdir: str):
    """
    Create a dataset by executing the C++ program.

    :param exec_path: The path to the executable that generates training data
    :param file_prefix: Prefix of the generated dataset(s)
    :param arguments: The program arguments (excluding log and file dirs)
    :param model_dir: The main directory of the model
    :param model_subdir: The relevant subdirectory inside model_dir
    """
    local_args = copy.deepcopy(arguments)
    local_args.extend([
        "--log",
        f"--file-prefix={file_prefix}",
        f"--model-dir={str(model_dir / model_subdir)}",
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


def rl_loop(data_dir: Path, max_iterations, exec_path: Path, dataset_args: list, train_config: dict,
            model_subdir: str, num_datasets_latest: int, num_datasets_recent: int, datasets_recent_include: float,
            rtpt: RTPT, model_init_dir: Optional[Path] = None):
    """
    The main RL loop which alternates between data generation and training:

    generation 0 -> training 0 & generation 1 -> training 1 & generation 2 -> ...

    :param data_dir: The directory of the run, all data and models will be saved there.
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
    :param model_init_dir: Directory of the model to be used in the first iteration. If None, a new model is created.
    """
    global stop_rl
    data_dir.mkdir(exist_ok=True, parents=True)

    # copy the executable and rename this copy according to rtpt (to be removed after the rl loop)
    # => if this process is killed, we don't have to manually rename the original executable again
    exec_copy_path = data_dir / exec_path.name
    if exec_copy_path.exists():
        exec_copy_path.unlink()
    shutil.copy2(str(exec_path.absolute()), str(exec_copy_path.absolute()))

    # The current iteration
    it = 0

    def print_it(msg: str):
        print(f"{datetime.now()} > It. {it}: {msg}")

    last_model_dir = None
    model_dir = data_dir / (str(it - 1) + "_model")

    # Before we can create a dataset, we need an initial model
    if model_init_dir is None:
        model_dir.mkdir(exist_ok=True)
        training.train_cnn.set_default_cuda_device(train_config)
        training.train_cnn.export_initial_model(train_config, model_dir)
        print("No initial model provided. Using new model.")
    else:
        assert model_init_dir.exists() and not is_empty(model_init_dir), f"Could not find model in '{model_init_dir}'!"
        shutil.copytree(str(model_init_dir), model_dir)
        rm_files_with_type(model_dir, ".trt")
        print("Using existing model.")

    # Loop: Train & Create -> Archive -> Train & Create -> ...
    train_future_res = None
    # The first iteration does not count, as we only generate a dataset
    while max_iterations < 0 or it <= max_iterations:
        model_dir = data_dir / (str(it - 1) + "_model")

        with stop_rl_lock:
            last_iteration = it == max_iterations or stop_rl

        if last_iteration:
            print_it("Entering the last iteration")

        # Start training if training data exists
        datasets = get_datatsets(data_dir)
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
            rtpt.step(subtitle=f"it={it:d}_of_{max_iterations:d}")

        # Create a new dataset using the current model
        if not last_iteration:
            # Create a new dataset
            print_it("Create dataset")

            exec_copy_path = rename_file(exec_copy_path, rtpt._get_title(), True)
            file_prefix = str(data_dir / f"{it}_data")
            sproc_create_dataset = create_dataset(exec_copy_path, file_prefix, dataset_args, model_dir, model_subdir)
            subprocess_verbose_wait(sproc_create_dataset)

            print_it("Dataset done")

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


def parse_extra_args(args_str: str):
    """
    Parses extra args and returns a dictionary with argument keys and string values (value can be None if not given).

    :param args_str: String of additional arguments of the form "--param1 --param2=value" or "param1 param2=value"
    :returns: exec arg dict
    """
    exec_args = dict()
    for arg_str in args_str.split(' '):
        if len(arg_str) == 0:
            continue

        if arg_str.startswith("--"):
            arg_str = arg_str[2:]

        key_value = arg_str.split('=')
        if len(key_value) == 1:
            # argument without value
            exec_args[key_value[0]] = None
        elif len(key_value) == 2:
            # key-value pair
            exec_args[key_value[0]] = key_value[1]
        else:
            raise ValueError(f"Illegal argument '{arg_str}'!")

    return exec_args


def can_cast_float(a) -> bool:
    """
    Checks if the given argument can be cast to float.

    :param a: any element, e.g. string, float, int..
    :returns: whether the given element a can be cast to float
    """
    try:
        float(a)
        return True
    except ValueError:
        return False


def try_cast(d: dict):
    """
    Casts elements of type string in a dictionary to boolean, float and int if they appear to be of these types.

    :param d: dictionary of strings
    :returns: the updated dictionary
    """
    d = d.copy()
    for key in d:
        val = d[key]
        if not isinstance(val, str):
            continue

        # check for float and int
        if can_cast_float(val):
            float_val = float(val)
            if float_val == int(float_val):
                # further cast it if this value is an integer
                d[key] = int(float_val)
            else:
                # otherwise, assign the float
                d[key] = float_val
            continue

        # check for boolean
        val_strip_lower = val.strip().lower()
        if val_strip_lower == 'true':
            d[key] = True
            continue
        if val_strip_lower == 'false':
            d[key] = False
            continue

    return d


def get_and_remove(d: dict, key, default):
    """
    Get an element from a dictionary and automatically remove it if it exists.

    :param d: the dictionary
    :param key: the key
    :param default: default value if d[key] does not exists
    """
    if key in d:
        val = d[key]
        del d[key]
        return val

    return default


def arg_dict_to_arg_list(d: dict):
    """
    Converts a dictionary to a program argument list.

    :param d: a dictionary of form {"param1": None, "param2": "value"}
    :returns: program argument list of the form ["--param1", "--param2=value"]
    """
    arg_list = []
    for key in d:
        val = d[key]
        if val is None:
            arg_list.append(f"--{key}")
        else:
            arg_list.append(f"--{key}={val}")
    return arg_list


def main():
    global stop_rl

    parser = argparse.ArgumentParser(description='PommerLearn RL Loop')
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument('--dir', default='.', type=check_dir,
                        help='The main training directory that is used to store all intermediate and archived results')
    parser.add_argument('--exec', default='./PommerLearn', type=check_file,
                        help='The path to the PommerLearn executable')
    parser.add_argument('--exec-args', default='', type=str,
                        help='Allows to forward arguments to the executable (and replaces default values). Provide '
                             'them as a string of the form "param1 param2=value" or "--param1 --param2=value"')
    parser.add_argument('--train-args', default='', type=str,
                        help='Allows to forward arguments to the trainer (and replaces default values). Provide '
                             'them as a string of the form "param1 param2=value" or "--param1 --param2=value"')
    parser.add_argument('--it', default=100, type=int,
                        help='Maximum number of iterations (-1 for endless run that has to be stopped manually)')
    parser.add_argument('--num-latest', default=4, type=int,
                        help='The number of last datasets that are used for training in each iteration')
    parser.add_argument('--num-recent', default=4, type=int,
                        help='The number of "recent" datasets that will be used in addition to --num-latest')
    parser.add_argument('--recent-include', default=0.1, type=float,
                        help='Defines the meaning of "recent" as a proportion of all datasets (e.g. 0.1 for last 10%%)')
    parser.add_argument('--name-initials', default='XX', type=str,
                        help='The name initials that are used to specify the user for the RTPT library.')
    parser.add_argument('--gpu', default=0 if torch.cuda.is_available() else None, type=int,
                        help='The device index for cuda (also passed to the executable for sample generation).')
    parser.add_argument('--model-init-dir', default=None, type=str,
                        help='Directory of the model to be used in the first iteration. '
                             'Creates new model if not specified.')
    parser.add_argument('--comment', default='', type=str,
                        help='Add a comment for easier run identification. Is appended to the run id and visible '
                             'in the archive and on tensorboard. Allowed characters: alphanumeric, underscore, dash')

    parsed_args = parser.parse_args()
    parsed_exec_args = parse_extra_args(parsed_args.exec_args)
    parsed_train_args = try_cast(parse_extra_args(parsed_args.train_args))

    if torch.cuda.is_available() and parsed_args.gpu is not None:
        device_str = f"cuda:{parsed_args.gpu}"
        # ensure that we can use that device before starting anything
        torch.zeros(1, device=device_str)
        # model_subdir = "torch_cuda"
        model_subdir = "onnx"
    else:
        device_str = "cpu"
        warnings.warn("Make sure that your executable is built for the correct device type. Only the gpu device index "
                      " (if available) is passed at the moment.")
        model_subdir = "torch_cpu"

    print(f"Using device '{device_str}'")

    run_id = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if len(parsed_args.comment) > 0:
        # create safe run id (=> filename) from the comment
        run_comment_name = re.sub(r'[^\w\-]+', '_', parsed_args.comment)
        run_id = f"{run_id}_{run_comment_name}"

    base_dir = Path(os.path.expanduser(parsed_args.dir))
    exec_path = Path(os.path.expanduser(parsed_args.exec))
    data_dir = base_dir / "archive" / run_id
    model_init_dir = Path(os.path.expanduser(parsed_args.model_init_dir)) if parsed_args.model_init_dir else None

    # Info: All path-related arguments should be set inside the rl loop

    train_config = {
        "device": device_str,
        "nb_epochs": 1,
        "only_test_last": False,
        "test_size": 0.1,
        "tensorboard_dir": str(base_dir / "runs" / run_id),
        "batches_until_eval": 100,
        # "train_sampling_mode": "weighted_value_class",
        # for lstm
        "use_flat_core": False,
        "use_lstm": False,
        "sequence_length": 8,
    }
    # make sure we don't overwrite some important settings
    parsed_train_args.pop("tensorboard_dir", None)
    parsed_train_args.pop("device", None)
    train_config.update(parsed_train_args)
    train_config = training.train_cnn.fill_default_config(train_config)

    dataset_args = [
        "--mode=ffa_mcts",
        f"--env-gen-seed-eps={get_and_remove(parsed_exec_args, 'env-gen-seed-eps', '2')}",
        "--max-games=-1",
        f"--targeted-samples={get_and_remove(parsed_exec_args, 'targeted-samples', '50000')}",
        "--state-size=0",
        f"--planning-agents={get_and_remove(parsed_exec_args, 'planning-agents', 'SimpleUnbiasedAgent')}",
        f"--simulations={get_and_remove(parsed_exec_args, 'simulations', '100')}",
        f"--movetime={get_and_remove(parsed_exec_args, 'movetime', '100')}",
    ]
    dataset_args.extend(arg_dict_to_arg_list(parsed_exec_args))

    pp = pprint.PrettyPrinter(indent=4)
    print("Train Config")
    pp.pprint(train_config)
    print("Dataset Args")
    pp.pprint(dataset_args)

    if torch.cuda.is_available() and parsed_args.gpu is not None:
        dataset_args += [f"--gpu={parsed_args.gpu}"]

    # Create RTPT object
    rtpt = RTPT(name_initials=parsed_args.name_initials, experiment_name='Pommer', max_iterations=parsed_args.it)

    # Start the RTPT tracking
    rtpt.start()

    # Start the rl loop
    rl_args = (data_dir, parsed_args.it, exec_path, dataset_args, train_config, model_subdir, parsed_args.num_latest,
               parsed_args.num_recent, parsed_args.recent_include, rtpt, model_init_dir)
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


if __name__ == "__main__":
    main()
