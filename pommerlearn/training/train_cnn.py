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
from pommerlearn.nn.a0_resnet import AlphaZeroResnet, init_weights
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def main():
    # ----------- HYPERPARAMETERS --------------
    train_config = {
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 1e-04,
        "value_loss_ratio": 0.01,
        "test_size": 0.2,
        "batch_size": 128,
        "random_state":  42,
        "nb_epochs":  10,
    }

    random_state = 42

    # Note: You have to create that dataset first
    f = zarr.open('data_0.zr', 'r')

    print("Info:")
    print(f.info)

    attrs = f.attrs.asdict()
    print("Attributes: {}".format(attrs.keys()))

    dataset_size = len(f['act'])
    actual_steps = attrs['Steps']
    print("Dataset size: {}, actual steps: {}".format(dataset_size, actual_steps))

    use_cuda = torch.cuda.is_available()

    train_loader, val_loader = prepare_dataset(f, train_config["test_size"], train_config["batch_size"], random_state)

    model = AlphaZeroResnet(num_res_blocks=3)
    init_weights(model)

    if use_cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=train_config["lr"], momentum=train_config["momentum"],
                          weight_decay=train_config["weight_decay"])

    policy_loss = nn.CrossEntropyLoss()
    value_loss = nn.MSELoss()

    run_training(model, train_config["nb_epochs"], optimizer, policy_loss, train_loader, use_cuda, val_loader, value_loss,
                 train_config["value_loss_ratio"])


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


def run_training(model, nb_epochs, optimizer, value_loss, policy_loss, value_loss_ratio, train_loader, val_loader,
                 use_cuda):
    """
    Trains a given model for a number of epochs
    :param model: Model to optimize
    :param nb_epochs: Number of epochs to train
    :param optimizer: Optimizer to use
    :param policy_loss: Policy loss object
    :param value_loss: Value loss object
    :param value_loss_ratio: Value loss ratio
    :param train_loader: Training data loader
    :param val_loader: Validation data loader
    :param use_cuda: True, when GPU should be used
    :return:
    """

    m_train = Metrics()
    global_step = 0
    writer = SummaryWriter()

    for epoch in range(nb_epochs):
        # training
        for batch_idx, (x_train, yv_train, yp_train) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            if use_cuda:
                x_train, yv_train, yp_train = x_train.cuda(), yv_train.cuda(), yp_train.cuda()
            x_train, yp_train = Variable(x_train), Variable(yp_train)
            combined_loss = update_metrics(m_train, model, policy_loss, value_loss, value_loss_ratio, x_train, yp_train,
                                           yv_train)

            combined_loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                m_val = get_val_loss(model, value_loss_ratio, value_loss, policy_loss, use_cuda, val_loader)
                print(f'epoch: {epoch}, batch index: {batch_idx + 1}, train value loss: {m_train.value_loss():5f},'
                      f' train policy loss: {m_train.policy_loss():5f}, train policy acc: {m_train.policy_acc():5f},'
                      f' train value loss: {m_train.value_loss():5f}, val value loss: {m_val.value_loss():5f},'
                      f' val policy loss: {m_train.policy_loss():5f}, val policy acc: {m_val.policy_acc():5f}')
                log_to_tensorboard(global_step, m_train, m_val, writer)
                m_train.reset()

            global_step += 1


def log_to_tensorboard(global_step, m_train, m_val, writer) -> None:
    """
    Logs all metrics to Tensorboard at the current global step
    :param global_step: Global step of batch update
    :param m_train: Metrics for training dataset
    :param m_val: Metrics for validation dataset
    :param writer: Tensorobard writer handle
    :return:
    """
    writer.add_scalars('Combined Loss', {'train': m_train.combined_loss(),
                                         'val': m_val.combined_loss()}, global_step)
    writer.add_scalars('Value Loss', {'train': m_train.value_loss(),
                                      'val': m_val.value_loss()}, global_step)
    writer.add_scalars('Policy Loss', {'train': m_train.policy_loss(),
                                       'val': m_val.policy_loss()}, global_step)
    writer.add_scalars('Policy Accuracy', {'train': m_train.policy_acc(),
                                           'val': m_val.policy_acc()}, global_step)


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
    :return:
    """
    value_out, policy_out = model(x_train)
    cur_policy_loss = policy_loss(policy_out, yp_train)
    cur_value_loss = value_loss(value_out, yv_train)
    _, pred_label = torch.max(policy_out.data, 1)
    metric.total_cnt += x_train.data.size()[0]
    metric.correct_cnt += float((pred_label == yp_train.data).sum())
    combined_loss = cur_policy_loss + value_loss_ratio * cur_value_loss
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


def prepare_dataset(f, test_size: float, batch_size: int, random_state: int) -> [DataLoader, DataLoader]:
    """
    Returns pytorch dataset loaders for a given zarr dataset object
    :param f: Loaded zarr dataset object
    :param test_size: Percentage of data to use for testing
    :param batch_size: Batch size to use for training
    :param random_state: Seed value for reproducibility
    :return: Training loader, Validation loader
    """
    x_train, x_val, yv_train, yv_val, yp_train, yp_val = train_test_split(f['obs'][:], f['val'][:], f['act'][:],
                                                                          test_size=test_size, random_state=random_state)
    x_train = torch.Tensor(x_train)
    yp_train = torch.Tensor(yp_train).long()
    yv_train = torch.Tensor(yv_train)
    x_val = torch.Tensor(x_val)
    yp_val = torch.Tensor(yp_val).long()
    yv_val = torch.Tensor(yv_val)
    data_train = TensorDataset(x_train, yv_train, yp_train)
    data_val = TensorDataset(x_val, yv_val, yp_val)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


if __name__ == '__main__':
    main()
