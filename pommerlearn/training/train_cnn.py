"""
@file: train_cnn.py
Created on 16.06.20
@project: PommerLearn
@author: queensgambit

Basic training script to replicate behaviour of baseline agent
"""
import os
from typing import Optional

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import onnx
from onnxsim import simplify

from data_augmentation import *
from nn import PommerModel
from nn.a0_resnet import AlphaZeroResnet, init_weights
from nn.rise_mobile_v3 import RiseV3
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from torch.optim.optimizer import Optimizer

from nn.simple_lstm import SimpleLSTM
from training.loss.masked_continuous_cross_entropy import MaskedContinuousCrossEntropyLoss
from training.loss.masked_mse import MaskedMSELoss
from training.lr_schedules.lr_schedules import CosineAnnealingSchedule, LinearWarmUp,\
    MomentumSchedule, OneCycleSchedule, ConstantSchedule
from dataset_util import create_data_loaders, log_dataset_stats, get_last_dataset_path
from training.metrics import Metrics
from training.train_util import rm_dir


def create_model(train_config):
    input_shape = (18, 11, 11)
    valid_models = ["a0", "risev3"]
    if train_config["model"] == "a0":
        model = AlphaZeroResnet(num_res_blocks=train_config["num_res_blocks"], nb_input_channels=input_shape[0], board_width=input_shape[2],
                                board_height=input_shape[1], act_type=train_config["act_type"],
                                channels_policy_head=train_config["channels_policy_head"], channels=train_config["channels"])
    elif train_config["model"] == "risev3":
        kernels = [3] * train_config["num_res_blocks"]
        se_types = [train_config["se_type"]] * train_config["num_res_blocks"]
        model = RiseV3(nb_input_channels=input_shape[0], board_width=input_shape[2], board_height=input_shape[1],
                       channels=train_config["channels"], channels_operating=train_config["channels_operating"], kernels=kernels, se_types=se_types, use_raw_features=False,
                       act_type=train_config["act_type"], use_downsampling=train_config["use_downsampling"],
                       channels_policy_head=train_config["channels_policy_head"],
                       slice_scalars=train_config["slice_scalars"],
                       # flat arguments
                       use_flat_core=train_config["use_flat_core"],
                       core_hidden=256, value_hidden=64, policy_hidden=128,
                       # .. with lstm
                       use_lstm=train_config["use_lstm"], lstm_layers=1
        )
    else:
        raise Exception(f'Invalid model "{train_config["model"]}" given. Valid models are "{valid_models}".')
    init_weights(model)

    return input_shape, model


def create_optimizer(model: nn.Module, train_config: dict):
    return optim.AdamW(model.parameters(), lr=train_config["max_lr"], weight_decay=train_config["weight_decay"])


def train_cnn(train_config):
    set_default_cuda_device(train_config)
    if "device" in train_config:
        device = train_config["device"]
        if isinstance(device, str):
            device = torch.device(device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Run training with device '{device}'")

    input_shape, model = create_model(train_config)
    model = model.to(device)

    train_sequence_length = train_config["sequence_length"] if model.is_stateful else None

    train_loader, val_loader = create_data_loaders(
        train_config["dataset_path"],
        train_config["value_version"], train_config["discount_factor"], train_config["mcts_val_weight"],
        train_config["test_size"], train_config["batch_size"], train_config["batch_size_test"],
        train_transform=train_config["dataset_train_transform"], sequence_length=train_sequence_length,
        num_workers=train_config["num_workers"], only_test_last=train_config["only_test_last"],
        train_sampling_mode=train_config["train_sampling_mode"], test_split_mode=train_config["test_split_mode"]
    )

    optimizer = create_optimizer(model, train_config)

    model_input_dir = None if train_config["torch_input_dir"] is None else Path(train_config["torch_input_dir"])
    if model_input_dir is not None:
        print(f"Loading torch state from {str(model_input_dir)}")
        load_torch_state(model, optimizer, str(get_torch_state_path(model_input_dir)), device)

    policy_loss = MaskedContinuousCrossEntropyLoss(train_config["policy_loss_argmax_target"])
    value_loss = MaskedMSELoss()

    total_it = len(train_loader) * train_config["nb_epochs"]
    lr_schedule, momentum_schedule = get_schedules(total_it, train_config)

    log_dir = train_config["tensorboard_dir"]
    comment = train_config["tensorboard_comment"]
    iteration = train_config["iteration"]

    log_dir = get_logdir(log_dir, comment)

    log_config(train_config, log_dir, iteration)

    # TODO: Maybe log data sets during RL loop instead?
    last_dataset_path = get_last_dataset_path(train_config["dataset_path"])
    log_dataset_stats(last_dataset_path, log_dir, iteration)

    global_step_start = train_config["global_step"]
    global_step_end = run_training(model, int(train_config["nb_epochs"]), optimizer, lr_schedule, momentum_schedule,
                                   value_loss, policy_loss, train_config["value_loss_ratio"],
                                   train_loader, val_loader, device, log_dir, global_step=global_step_start,
                                   batches_until_eval=train_config["batches_until_eval"])

    base_dir = Path(train_config["output_dir"])
    batch_sizes = train_config["model_batch_sizes"]
    export_model(model, batch_sizes, input_shape, base_dir)
    save_torch_state(model, optimizer, str(get_torch_state_path(base_dir)))

    result_dict = {
        "global_step": global_step_end
    }

    return result_dict


def get_logdir(log_dir, comment):
    if log_dir is not None:
        return log_dir

    # create new log_dir and append comment like in SummaryWriter
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    return os.path.join('runs', current_time + '_' + socket.gethostname() + comment)


def get_schedules(total_it, train_config):
    """
    Returns a learning rate and momentum schedule

    :param total_it: Total iterations
    :param train_config: Training configuration dictionary
    """
    if train_config["schedule"] == "cosine_annealing":
        lr_schedule = CosineAnnealingSchedule(train_config["min_lr"], train_config["max_lr"], total_it * 0.7)
        lr_schedule = LinearWarmUp(lr_schedule, start_lr=0, length=total_it * 0.25)
    elif train_config["schedule"] == "one_cycle":
        lr_schedule = OneCycleSchedule(start_lr=train_config["max_lr"] / 8, max_lr=train_config["max_lr"],
                                       cycle_length=total_it * .3, cooldown_length=total_it * .6,
                                       finish_lr=train_config["min_lr"])
        lr_schedule = LinearWarmUp(lr_schedule, start_lr=train_config["min_lr"], length=total_it / 30)
    elif train_config["schedule"] == "constant":
        lr_schedule = ConstantSchedule(train_config["max_lr"])
    else:
        raise Exception(f"Invalid schedule type '{train_config['schedule']}' given.")
    momentum_schedule = MomentumSchedule(lr_schedule, train_config["min_lr"], train_config["max_lr"],
                                         train_config["min_momentum"], train_config["max_momentum"])

    return lr_schedule, momentum_schedule


def get_torch_state_path(base_dir: Path) -> Path:
    return base_dir / "torch_state.tar"


def load_torch_state(model: nn.Module, optimizer: Optimizer, path: str, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def save_torch_state(model: nn.Module, optimizer: Optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def export_initial_model(train_config, base_dir: Path):
    input_shape, model = create_model(train_config)
    optimizer = create_optimizer(model, train_config)

    export_model(model, train_config["model_batch_sizes"], input_shape, base_dir, torch_cpu=True,
                 torch_cuda=torch.cuda.is_available(), onnx=True)
    save_torch_state(model, optimizer, str(get_torch_state_path(base_dir)))


def export_model(model, batch_sizes, input_shape, dir=Path('.'), torch_cpu=True, torch_cuda=True, onnx=True,
                 verbose=False):
    """
    Exports the model in ONNX and Torch Script Module.

    :param model: Pytorch model
    :param batch_sizes: List of batch sizes to use for export
    :param input_shape: Input shape of the model
    :param dir: The base path for all models
    :param torch_cpu: Whether to export as script module with cpu inputs
    :param torch_cuda: Whether to export as script module with cuda inputs
    :param onnx: Whether to export as onnx
    :param verbose: Print debug information
    """

    if dir.exists():
        # make sure that all the content is deleted first so we don't run into strange caching issues
        rm_dir(dir, keep_empty_dir=False)

    dir.mkdir(parents=True, exist_ok=False)

    onnx_dir = dir / "onnx"
    if torch_cpu:
        onnx_dir.mkdir(parents=True, exist_ok=False)

    cpu_dir = dir / "torch_cpu"
    if torch_cpu:
        cpu_dir.mkdir(parents=True, exist_ok=False)

    torch_cuda = torch_cuda and torch.cuda.is_available()
    cuda_dir = dir / "torch_cuda"
    if torch_cuda:
        cuda_dir.mkdir(parents=True, exist_ok=False)

    model = model.eval()

    for batch_size in batch_sizes:
        dummy_input = torch.ones(batch_size, input_shape[0], input_shape[1], input_shape[2], dtype=torch.float)

        if model.is_stateful:
            dummy_input = model.flatten(dummy_input, model.get_init_state_bf_flat(batch_size, "cpu"))
            model.set_input_options(sequence_length=None, has_state_input=True)
        else:
            dummy_input = model.flatten(dummy_input, None)
            model.set_input_options(sequence_length=None, has_state_input=False)

        if onnx:
            dummy_input = dummy_input.cpu()
            model = model.cpu()
            export_to_onnx(model, batch_size, dummy_input, onnx_dir)

        if torch_cpu:
            dummy_input = dummy_input.cpu()
            model = model.cpu()
            export_as_script_module(model, batch_size, dummy_input, cpu_dir)

        if torch_cuda:
            dummy_input = dummy_input.cuda()
            model = model.cuda()
            export_as_script_module(model, batch_size, dummy_input, cuda_dir)

        if verbose:
            print("Input shape: ")
            print(dummy_input.shape)
            print("Output shape: ")
            for i, e in enumerate(model(dummy_input)):
                print(f"{i}: {e.shape}")


def export_to_onnx(model, batch_size, dummy_input, dir) -> None:
    """
    Exports the model to ONNX format to allow later import in TensorRT.

    :param model: Pytorch model
    :param batch_size: The batch size of the input
    :param dummy_input: Dummy input which defines the input shape for the model
    :return:
    """
    if model.is_stateful:
        input_names = ["data"]
        output_names = ["value_out", "policy_out", "auxiliary_out"]
    else:
        input_names = ["data"]
        output_names = ["value_out", "policy_out"]

    model_filepath = str(dir / Path(f"model-bsize-{batch_size}.onnx"))
    torch.onnx.export(model, dummy_input, model_filepath, input_names=input_names,
                      output_names=output_names)
    # simplify ONNX model
    # https://github.com/daquexian/onnx-simplifier
    model = onnx.load(model_filepath)
    model_simp, check = simplify(model)
    onnx.save(model_simp, model_filepath)
    if not check:
        raise Exception("Simplified ONNX model could not be validated")


def export_as_script_module(model, batch_size, dummy_input, dir) -> None:
    """
    Exports the model to a Torch Script Module to allow later import in C++.

    :param model: Pytorch model
    :param batch_size: The batch size of the input
    :param dummy_input: Dummy input which defines the input shape for the model
    :return:
    """

    # generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, dummy_input)

    # serialize script module to file
    traced_script_module.save(str(dir / Path(f"model-bsize-{batch_size}.pt")))


def run_training(model: PommerModel, nb_epochs, optimizer, lr_schedule, momentum_schedule, value_loss, policy_loss,
                 value_loss_ratio, train_loader, val_loader, device, log_dir, global_step=0,
                 batches_until_eval: Optional[int] = 100):
    """
    Trains a given model for a number of epochs

    :param model: Model to optimize
    :param nb_epochs: Number of epochs to train
    :param optimizer: Optimizer to use
    :param lr_schedule: LR-scheduler
    :param momentum_schedule: Momentum scheduler
    :param policy_loss: Policy loss object
    :param value_loss: Value loss object
    :param value_loss_ratio: Value loss ratio
    :param train_loader: Training data loader
    :param val_loader: Validation data loader (ignored if None)
    :param device: The device that should be used for training
    :param log_dir: Tensorboard log_dir
    :param global_step: The global step used for logging
    :param batches_until_eval: Number of batches between evaluations of the current model. The model will also be
                               evaluated at the end.
    :return: global_step after training
    """

    m_train = Metrics()
    local_step = 0

    writer_train = SummaryWriter(log_dir=log_dir)
    if val_loader is not None:
        writer_val = SummaryWriter(log_dir=log_dir + "-val")

    # TODO: Nested progress bars would be ideal
    progress = tqdm(total=len(train_loader) * nb_epochs, smoothing=0)

    for epoch in range(nb_epochs):
        # training
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if model.is_stateful:
                ids, x_train, yv_train, ya_train, yp_train = batch
            else:
                ids = None
                x_train, yv_train, ya_train, yp_train = batch

            x_train = x_train.to(device)
            yv_train = yv_train.to(device)
            ya_train = ya_train.to(device)
            yp_train = yp_train.to(device)

            if global_step == 0:
                # TODO: move outside training loop
                if model.is_stateful:
                    model.set_input_options(sequence_length=x_train[0].shape[0], has_state_input=True)
                    init_state = model.get_init_state_bf_flat(1, device=device)
                    writer_train.add_graph(model, model.flatten(x_train[0][None], init_state))
                else:
                    model.set_input_options(sequence_length=None, has_state_input=False)
                    writer_train.add_graph(model, model.flatten(x_train[0][None], None))

            model.train()
            if model.is_stateful:
                # Assumption: We always train stateful models with sequences and training data has the shape
                #             (batch dim, sequence dim, data)
                # Important: We do not use the state input in training.
                model.set_input_options(sequence_length=x_train.shape[1], has_state_input=False)
            else:
                model.set_input_options(sequence_length=None, has_state_input=False)

            combined_loss, _ = m_train.update(model, policy_loss, value_loss, value_loss_ratio, x_train, yv_train,
                                           ya_train, yp_train, ids=ids, device=device)

            combined_loss.backward()
            lr, momentum = 0, 0
            for param_group in optimizer.param_groups:
                lr = lr_schedule(local_step)
                param_group['lr'] = lr

                momentum = momentum_schedule(local_step)
                param_group['momentum'] = momentum

            optimizer.step()

            if (batches_until_eval is not None and (batch_idx + 1) % batches_until_eval == 0) \
                    or (batch_idx + 1) == len(train_loader):
                msg = f' epoch: {epoch}, batch index: {batch_idx + 1}, train value loss: {m_train.value_loss():5f},'\
                      f' train policy loss: {m_train.policy_loss():5f}, train policy acc: {m_train.policy_acc():5f}'

                # log last lr and momentum
                writer_train.add_scalar('Hyperparameter/Learning Rate', lr, global_step)
                writer_train.add_scalar('Hyperparameter/Momentum', momentum, global_step)

                # log aggregated train stats
                m_train.log_to_tensorboard(writer_train, global_step)
                m_train.reset()

                if val_loader is not None:
                    model.eval()
                    if model.is_stateful:
                        m_val = get_stateful_val_loss(model, value_loss_ratio, value_loss, policy_loss, device,
                                                      val_loader)
                    else:
                        m_val = get_val_loss(model, value_loss_ratio, value_loss, policy_loss, device, val_loader)

                    m_val.log_to_tensorboard(writer_val, global_step)
                    msg += f', val value loss: {m_val.value_loss():5f}, val policy loss: {m_val.policy_loss():5f},'\
                           f' val policy acc: {m_val.policy_acc():5f}'

                print(msg)

            global_step += 1
            local_step += 1
            progress.update(1)

    progress.close()

    writer_train.close()
    if val_loader is not None:
        writer_val.close()

    # make sure to empty cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return global_step


def get_val_loss(model, value_loss_ratio, value_loss, policy_loss, device, data_loader) -> Metrics:
    """
    Returns the validation metrics by evaluating it on the full validation dataset

    :param model: Model to evaluate
    :param value_loss_ratio: Value loss ratio
    :param value_loss: Value loss object
    :param policy_loss: Policy loss object
    :param device: The device that is used
    :param data_loader: Data loader object (e.g. val_loader)
    :return: Updated metric object
    """

    m_val = Metrics()

    for batch_idx, (x, yv_val, ya_val, yp_val) in enumerate(data_loader):
        x = x.to(device)
        yv_val = yv_val.to(device)
        ya_val = ya_val.to(device)
        yp_val = yp_val.to(device)

        x, ya_val = Variable(x, requires_grad=False), Variable(ya_val, requires_grad=False)

        model.set_input_options(sequence_length=None, has_state_input=False)
        m_val.update(model, policy_loss, value_loss, value_loss_ratio, x, yv_val, ya_val, yp_val)

    return m_val


def get_stateful_val_loss(model, value_loss_ratio, value_loss, policy_loss, device, data_loader,
                 verbose = False) -> Metrics:
    """
    Returns the validation metrics by evaluating it on the full validation dataset

    :param model: Model to evaluate
    :param value_loss_ratio: Value loss ratio
    :param value_loss: Value loss object
    :param policy_loss: Policy loss object
    :param device: The device that is used
    :param data_loader: Data loader object (e.g. val_loader)
    :param verbose: Whether to output debugging info
    :return: Updated metric object
    """

    m_val = Metrics()

    current_episode_id = None

    for batch_idx, (ids, obs, val, act, pol) in enumerate(data_loader):
        if verbose:
            print("Got test batch with ids: ", ids)

        if current_episode_id is None or current_episode_id != ids[0]:
            # the first sample from this batch is from a new episode, reset the state
            model_state = model.get_init_state_bf_flat(1, device)

        if current_episode_id is None:
            current_episode_id = ids[0]

        current_idx = 0
        while current_idx < len(ids):
            current_episode_id = ids[current_idx]

            if ids[-1] == current_episode_id:
                # we can process the batch until the end
                until_idx = len(ids)
                reset_state = False
            else:
                # we have to check when the next episode begins
                for i in range(current_idx, len(ids)):
                    if ids[i] != current_episode_id:
                        until_idx = i
                        break

                # and reset the state after these samples
                reset_state = True

            obs_part = obs[current_idx:until_idx].to(device)
            val_part = val[current_idx:until_idx].to(device)
            act_part = act[current_idx:until_idx].to(device)
            pol_part = pol[current_idx:until_idx].to(device)

            # transform batch to sequence
            obs_part = obs_part.unsqueeze(0)
            val_part = val_part.unsqueeze(0)
            act_part = act_part.unsqueeze(0)
            pol_part = pol_part.unsqueeze(0)

            if verbose:
                print(f"Prepared data for range {current_idx} until {until_idx}")

            # update loss & get next state
            model.set_input_options(sequence_length=(until_idx - current_idx), has_state_input=True)
            loss, next_state = m_val.update(model, policy_loss, value_loss, value_loss_ratio, obs_part, val_part,
                                            act_part, pol_part, model_state=model_state)

            current_idx = until_idx
            if reset_state:
                model_state = model.get_init_state_bf_flat(1, device)
                if verbose:
                    print("Reset state")
            else:
                model_state = next_state
                if verbose:
                    print("Use next state")

    return m_val


def set_default_cuda_device(train_config):
    """
    Sets the default cuda device for pytorch according to the config.

    :param train_config: The train config dictionary
    """
    if torch.cuda.is_available() and "device" in train_config and train_config["device"]:
        torch.cuda.set_device(train_config["device"])


def log_config(train_config, log_dir, iteration):
    """
    Log the train config to tensorboard.

    :param train_config: The train config dictionary
    :param log_dir: The logdir of the summary writer
    :param iteration: The iteration this config belongs to
    """

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text(f"Train Config", str(train_config), iteration)
    writer.close()


def fill_default_config(train_config):
    default_data_transform = RandomTransform(
        Identity(),
        Rotate90(1),
        Rotate90(2),
        Rotate90(3),
        FlipX(),
        ComposeTransform(FlipX(), Rotate90(1)),
        ComposeTransform(FlipX(), Rotate90(2)),
        ComposeTransform(FlipX(), Rotate90(3)),
    )

    default_config = {
        # input
        # dataset_path: The path information of the zarr dataset(s) which will be used. Should be a single string
        # (of a path) or a list containing a) strings (paths) or b) tuples of the form (path, proportion) where
        # 0 <= proportion <= 1 is the proportion of total samples which will be selected from this data set.
        # Examples: "data_0.zr", [("data_0.zr", 0.25), ("data_1.zr", 0.5), "data_2.zr"]
        "dataset_path": "data_0.zr",  # "2020/mcts_data_500.zr"
        "dataset_train_transform": default_data_transform,  # None
        "torch_input_dir": None,
        # output
        "output_dir": "./model",
        "model_batch_sizes": [1, 8],
        # hyperparameters
        "value_version": 1,
        "discount_factor": 0.9,
        "mcts_val_weight": 0.5,  # None or in [0, 1]
        "train_sampling_mode": "complete",  # "complete", "weighted_steps_to_end", "weighted_actions"
        "policy_loss_argmax_target": False,
        "min_lr": 0.0001,
        "max_lr": 0.05,
        "min_momentum": 0.8,
        "max_momentum": 0.95,
        "schedule": "one_cycle",  # "cosine_annealing", "one_cycle", "constant"
        "momentum": 0.9,
        "weight_decay": 1e-03,
        "value_loss_ratio": 0.1,
        "test_size": 0.2,
        "test_split_mode": "simple",
        "only_test_last": False,
        "batch_size": 128,  # warning: should be adapted when using sequences
        "batch_size_test": 1024,
        "random_state":  42,
        "nb_epochs":  20,
        "model": "risev3",  # "a0", "risev3", "lstm"
        "sequence_length": 8,  # only used when model is stateful
        "use_downsampling": True,
        "se_type": None,  # None, "se", "cbam", "eca_se", "ca_se", "cm_se", "sa_se", "sm_se"
        "num_res_blocks": 4,
        "act_type": "relu",
        "channels_policy_head": 16,
        "channels": 64,
        "channels_operating": 256,
        "use_flat_core": False,
        "slice_scalars": False,  # only with flat core
        "use_lstm": False,  # only with flat core
        # logging
        "tensorboard_dir": None,  # None means tensorboard will create a unique path for the run
        "tensorboard_comment": '',  # Tensorboard comment argument
        "iteration": 0,
        "global_step": 0,
        "batches_until_eval": 100,
        # training
        "num_workers": 4
    }

    for key in train_config:
        default_config[key] = train_config[key]

    return default_config


if __name__ == '__main__':
    train_cnn(fill_default_config({}))
