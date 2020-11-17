"""
@file: train_cnn.py
Created on 16.06.20
@project: PommerLearn
@author: queensgambit

Basic training script to replicate behaviour of baseline agent
"""

import zarr
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from nn.a0_resnet import AlphaZeroResnet, init_weights
from nn.rise_mobile_v3 import RiseV3
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.optim.optimizer import Optimizer
from training.lr_schedules.lr_schedules import CosineAnnealingSchedule, plot_schedule, LinearWarmUp,\
    MomentumSchedule, OneCycleSchedule, ConstantSchedule


def create_model(train_config):
    input_shape = (18, 11, 11)
    valid_models = ["a0", "risev3"]
    if train_config["model"] == "a0":
        model = AlphaZeroResnet(num_res_blocks=3, nb_input_channels=input_shape[0], board_width=input_shape[1],
                                board_height=input_shape[2])
    elif train_config["model"] == "risev3":
        kernels = [[3]] * 3
        se_types = [None] * 3
        model = RiseV3(nb_input_channels=input_shape[0], board_width=input_shape[1], board_height=input_shape[2],
                       kernels=kernels, se_types=se_types, use_raw_features=True, act_type="relu")
    else:
        raise Exception(f'Invalid model "{train_config["model"]}" given. Valid models are "{valid_models}".')
    init_weights(model)

    return input_shape, model


def create_optimizer(model: nn.Module, train_config: dict):
    return optim.SGD(model.parameters(), lr=train_config["max_lr"], momentum=train_config["max_momentum"],
                     weight_decay=train_config["weight_decay"])


def train_cnn(train_config):
    z = zarr.open(train_config["dataset_path"], 'r')
    z_samples = z.attrs["Steps"]

    print(f"Opened dataset with {z_samples} samples from {len(z.attrs['EpisodeSteps'])} episodes")

    use_cuda = torch.cuda.is_available()
    print(f"CUDA enabled: {use_cuda}")

    train_loader, val_loader = prepare_dataset(z, train_config["test_size"], train_config["batch_size"],
                                               train_config["random_state"])

    input_shape, model = create_model(train_config)

    if use_cuda:
        model = model.cuda()

    optimizer = create_optimizer(model, train_config)

    model_input_dir = None if train_config["torch_input_dir"] is None else Path(train_config["torch_input_dir"])
    if model_input_dir is not None:
        print(f"Loading torch state from {str(model_input_dir)}")
        load_torch_state(model, optimizer, str(get_torch_state_path(model_input_dir)))

    policy_loss = nn.CrossEntropyLoss()
    value_loss = nn.MSELoss()

    total_it = len(train_loader) * train_config["nb_epochs"]
    lr_schedule, momentum_schedule = get_schedules(total_it, train_config)

    log_dir = train_config["tensorboard_dir"]
    iteration = train_config["iteration"]

    log_config(train_config, log_dir, iteration)
    log_dataset_stats(z, log_dir, iteration)

    global_step_start = train_config["global_step"]
    global_step_end = run_training(model, train_config["nb_epochs"], optimizer, lr_schedule, momentum_schedule,
                                   value_loss, policy_loss, train_config["value_loss_ratio"], train_loader, val_loader,
                                   use_cuda, log_dir, global_step=global_step_start)

    base_dir = Path(train_config["output_dir"])
    batch_sizes = train_config["model_batch_sizes"]
    export_model(model, batch_sizes, input_shape, base_dir)
    save_torch_state(model, optimizer, str(get_torch_state_path(base_dir)))

    result_dict = {
        "global_step": global_step_end
    }

    return result_dict


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


def load_torch_state(model: nn.Module, optimizer: Optimizer, path: str):
    checkpoint = torch.load(path)
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

    export_model(model, train_config["model_batch_sizes"], input_shape, base_dir)
    save_torch_state(model, optimizer, str(get_torch_state_path(base_dir)))


def export_model(model, batch_sizes, input_shape, dir=Path('.'), torch_cpu=True, torch_cuda=True, onnx=True):
    """
    Exports the model in ONNX and Torch Script Module.

    :param model: Pytorch model
    :param batch_sizes: List of batch sizes to use for export
    :param input_shape: Input shape of the model
    :param use_cuda: Whether cuda is enabled
    :param dir: The base path for all models
    :param torch_cpu: Whether to export as script module with cpu inputs
    :param torch_cuda: Whether to export as script module with cuda inputs
    :param onnx: Whether to export as onnx
    """
    dir.mkdir(parents=True, exist_ok=True)

    onnx_dir = dir / "onnx"
    if torch_cpu:
        onnx_dir.mkdir(parents=True, exist_ok=True)

    cpu_dir = dir / "torch_cpu"
    if torch_cpu:
        cpu_dir.mkdir(parents=True, exist_ok=True)

    torch_cuda = torch_cuda and torch.cuda.is_available()
    cuda_dir = dir / "torch_cuda"
    if torch_cuda:
        cuda_dir.mkdir(parents=True, exist_ok=True)

    for batch_size in batch_sizes:
        dummy_input = torch.ones(batch_size, input_shape[0], input_shape[1], input_shape[2], dtype=torch.float)

        if onnx:
            dummy_input = dummy_input.cpu()
            model = model.cpu()
            export_to_onnx(model, dummy_input, onnx_dir)

        if torch_cpu:
            dummy_input = dummy_input.cpu()
            model = model.cpu()
            export_as_script_module(model, dummy_input, cpu_dir)

        if torch_cuda:
            dummy_input = dummy_input.cuda()
            model = model.cuda()
            export_as_script_module(model, dummy_input, cuda_dir)


def export_to_onnx(model, dummy_input, dir) -> None:
    """
    Exports the model to ONNX format to allow later import in TensorRT.
    :param model: Pytorch model
    :param dummy_input: Dummy input which defines the input shape for the model
    :return:
    """
    input_names = ["data"]
    output_names = ["value_out", "policy_out"]
    torch.onnx.export(model, dummy_input, str(dir / Path(f"model-bsize-{dummy_input.size(0)}.onnx")), input_names=input_names,
                      output_names=output_names)


def export_as_script_module(model, dummy_input, dir) -> None:
    """
    Exports the model to a Torch Script Module to allow later import in C++.
    :param model: Pytorch model
    :param dummy_input: Dummy input which defines the input shape for the model
    :return:
    """
    # generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, dummy_input)

    # serialize script module to file
    traced_script_module.save(str(dir / Path(f"model-bsize-{dummy_input.size(0)}.pt")))


class Metrics:
    """
    Class which stores metric attributes as a struct
    """
    def __init__(self):
        self.correct_cnt = 0
        self.total_cnt = 0
        self.sum_combined_loss = 0
        self.sum_policy_loss = 0
        self.sum_value_loss = 0
        self.steps = 0

    def reset(self):
        self.correct_cnt = 0
        self.total_cnt = 0
        self.sum_combined_loss = 0
        self.sum_policy_loss = 0
        self.sum_value_loss = 0
        self.steps = 0

    def combined_loss(self):
        return self.sum_combined_loss / self.steps

    def value_loss(self):
        return self.sum_value_loss / self.steps

    def policy_loss(self):
        return self.sum_policy_loss / self.steps

    def policy_acc(self):
        return self.correct_cnt / self.total_cnt


def run_training(model, nb_epochs, optimizer, lr_schedule, momentum_schedule, value_loss, policy_loss, value_loss_ratio, train_loader, val_loader,
                 use_cuda, log_dir, global_step=0):
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
    :param use_cuda: True, when GPU should be used
    :param log_dir: The (base) log dir for the tensorboard writer(s)
    :param global_step: The global step used for logging
    :return:
    """

    m_train = Metrics()
    local_step = 0

    writer_train = SummaryWriter(log_dir=log_dir)
    if val_loader is not None:
        log_dir_val = None if log_dir is None else log_dir + "-val"
        writer_val = SummaryWriter(log_dir=log_dir_val, comment='-val')

    exported_graph = False

    # TODO: Nested progress bars would be ideal
    progress = tqdm(total=len(train_loader) * nb_epochs, smoothing=0)

    for epoch in range(nb_epochs):
        # training
        for batch_idx, (x_train, yv_train, yp_train) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                x_train, yv_train, yp_train = x_train.cuda(), yv_train.cuda(), yp_train.cuda()

            if not exported_graph:
                writer_train.add_graph(model, x_train)
                exported_graph = True

            x_train, yp_train = Variable(x_train), Variable(yp_train)
            combined_loss = update_metrics(m_train, model, policy_loss, value_loss, value_loss_ratio, x_train, yp_train,
                                           yv_train)

            combined_loss.backward()
            for param_group in optimizer.param_groups:
                lr = lr_schedule(local_step)
                writer_train.add_scalar('Hyperparameter/Learning Rate', lr, global_step)
                param_group['lr'] = lr

                momentum = momentum_schedule(local_step)
                writer_train.add_scalar('Hyperparameter/Momentum', momentum, global_step)
                param_group['momentum'] = momentum

            optimizer.step()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                msg = f' epoch: {epoch}, batch index: {batch_idx + 1}, train value loss: {m_train.value_loss():5f},'\
                      f' train policy loss: {m_train.policy_loss():5f}, train policy acc: {m_train.policy_acc():5f},'\
                      f' train value loss: {m_train.value_loss():5f}'

                # TODO: Log metric in every step
                log_metrics(writer_train, m_train, global_step)
                m_train.reset()

                if val_loader is not None:
                    # print validation stats
                    m_val = get_val_loss(model, value_loss_ratio, value_loss, policy_loss, use_cuda, val_loader)
                    log_metrics(writer_val, m_val, global_step)
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

    return global_step


def log_metrics(writer, metrics, global_step) -> None:
    """
    Logs all metrics to Tensorboard at the current global step.

    :param global_step: Global step of batch update
    :param metrics: Metrics which shall be logged
    :param writer: Tensorobard writer handle
    :return:
    """

    writer.add_scalar('Loss/Value', metrics.value_loss(), global_step)
    writer.add_scalar('Loss/Policy', metrics.policy_loss(), global_step)
    writer.add_scalar('Loss/Combined', metrics.combined_loss(), global_step)

    writer.add_scalar('Policy Accuracy', metrics.policy_acc(), global_step)

    writer.flush()


def update_metrics(metric, model, policy_loss, value_loss, value_loss_ratio, x_train, yp_train, yv_train):
    """
    Updates the metrics and calculates the combined loss
    :param metric: Metric object
    :param model: Model to optimize
    :param policy_loss: Policy loss object
    :param value_loss: Value loss object
    :param value_loss_ratio: Value loss ratio
    :param x_train: Training samples
    :param yp_train: Target labels for policy
    :param yv_train: Target labels for value
    :return: The combined loss
    """

    # get the combined loss
    value_out, policy_out = model(x_train)
    cur_policy_loss = policy_loss(policy_out, yp_train)
    cur_value_loss = value_loss(value_out.view(-1), yv_train)
    combined_loss = (1 - value_loss_ratio) * cur_policy_loss + value_loss_ratio * cur_value_loss

    # update metrics
    _, pred_label = torch.max(policy_out.data, 1)
    metric.total_cnt += x_train.data.size()[0]
    metric.correct_cnt += float((pred_label == yp_train.data).sum())
    metric.sum_policy_loss += float(cur_policy_loss.data)
    metric.sum_value_loss += float(cur_value_loss.data)
    metric.sum_combined_loss += float(combined_loss.data)
    metric.steps += 1

    return combined_loss


def get_val_loss(model, value_loss_ratio, value_loss, policy_loss, use_cuda, data_loader) -> Metrics:
    """
    Returns the validation metrics by evaluating it on the full validation dataset
    :param model: Model to evaluate
    :param value_loss_ratio: Value loss ratio
    :param value_loss: Value loss object
    :param policy_loss: Policy loss object
    :param use_cuda: Boolean whether GPU is used
    :param data_loader: Data loader object (e.g. val_loader)
    :return: Updated metric object
    """

    m_val = Metrics()

    for batch_idx, (x, yv_val, yp_val) in enumerate(data_loader):
        if use_cuda:
            x, yv_val, yp_val = x.cuda(), yv_val.cuda(), yp_val.cuda()
        x, yp_val = Variable(x, requires_grad=False), Variable(yp_val, requires_grad=False)
        update_metrics(m_val, model, policy_loss, value_loss, value_loss_ratio, x, yp_val, yv_val)
    return m_val


def prepare_dataset(z, test_size: float, batch_size: int, random_state: int) -> [DataLoader, DataLoader]:
    """
    Returns pytorch dataset loaders for a given zarr dataset object
    :param z: Loaded zarr dataset object
    :param test_size: Percentage of data to use for testing
    :param batch_size: Batch size to use for training
    :param random_state: Seed value for reproducibility
    :return: Training loader, Validation loader
    """

    z_steps = z.attrs["Steps"]
    obs = z['obs'][:z_steps]
    val = z['val'][:z_steps]
    act = z['act'][:z_steps]

    def get_loader(x, yv, yp):
        x = torch.Tensor(x)
        yv = torch.Tensor(yv)
        yp = torch.Tensor(yp).long()
        return DataLoader(TensorDataset(x, yv, yp), batch_size=batch_size, shuffle=True)

    if test_size == 0:
        return get_loader(obs, val, act), None

    if 0 < test_size < 1:
        x_train, x_val, yv_train, yv_val, yp_train, yp_val = train_test_split(
            obs, val, act,
            test_size=test_size, random_state=random_state
        )

        return get_loader(x_train, yv_train, yp_train), get_loader(x_val, yv_val, yp_val)

    raise ValueError(f"Incorrect test size: {test_size}")


def log_dataset_stats(z, log_dir, iteration):
    """
    Log dataset stats to tensorboard.

    :param z: The zarr dataset
    :param log_dir: The logdir of the summary writer
    :param iteration: The iteration this dataset belongs to
    """
    writer = SummaryWriter(log_dir=log_dir)

    steps = np.array(z.attrs["EpisodeSteps"])
    winners = np.array(z.attrs["EpisodeWinner"])
    done = np.array(z.attrs["EpisodeDone"])
    actions = z.attrs["EpisodeActions"]

    num_episodes = len(steps)

    writer.add_scalar("Dataset/Episodes", num_episodes, iteration)
    writer.add_scalar("Dataset/Steps mean", steps.mean(), iteration)
    writer.add_scalar("Dataset/Steps std", steps.std(), iteration)

    for a in range(0, 4):
        winner_a = np.sum(winners[:] == a)
        writer.add_scalar(f"Dataset/Win ratio {a}", winner_a / num_episodes, iteration)

        actions_a = []
        episode_steps = np.empty(len(actions))
        for i, ep in enumerate(actions):
            ep_actions = ep[a]
            episode_steps[i] = len(ep_actions)
            actions_a += ep_actions

        writer.add_scalar(f"Dataset/Steps mean {a}", episode_steps.mean(), iteration)

        # TODO: Correct bin borders
        writer.add_histogram(f"Dataset/Actions {a}", np.array(actions_a), iteration)

    no_winner = np.sum((winners == -1) * (done == True))
    writer.add_scalar(f"Dataset/Draw ratio", no_winner / num_episodes, iteration)

    not_done = np.sum(done == False)
    writer.add_scalar(f"Dataset/Not done ratio", not_done / num_episodes, iteration)

    writer.close()


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
    default_config = {
        # input
        "dataset_path": "data_0.zr",
        "torch_input_dir": None,
        # output
        "output_dir": "./model",
        "model_batch_sizes": [1, 8],
        # hyperparameters
        "min_lr": 0.0001,
        "max_lr": 0.05,
        "min_momentum": 0.8,
        "max_momentum": 0.95,
        "schedule": "one_cycle",  # "cosine_annealing", "one_cycle", "constant"
        "momentum": 0.9,
        "weight_decay": 1e-03,
        "value_loss_ratio": 0.01,
        "test_size": 0.2,
        "batch_size": 128,
        "random_state":  42,
        "nb_epochs":  10,
        "model": "a0",  # "a0", "risev3"
        # logging
        "tensorboard_dir": None,  # None means tensorboard will create a unique path for the run
        "iteration": 0,
        "global_step": 0,
    }

    for key in train_config:
        default_config[key] = train_config[key]

    return default_config


if __name__ == '__main__':
    train_cnn(fill_default_config({}))
