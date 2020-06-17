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
import sys
sys.path.append("../../")
from pommerlearn.nn.a0_resnet import AlphaZeroResnet, init_weights
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def main():
    # ----------- HYPERPARAMETERS --------------
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-04
    value_loss_ratio = 0.01
    test_size = 0.2
    batch_size = 128
    random_state = 42
    nb_epochs = 10

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

    train_loader, val_loader = prepare_dataset(f, test_size, batch_size, random_state)

    model = AlphaZeroResnet(num_res_blocks=3)
    init_weights(model)

    if use_cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    policy_loss = nn.CrossEntropyLoss()
    value_loss = nn.MSELoss()

    run_training(model, nb_epochs, optimizer, policy_loss, train_loader, use_cuda, val_loader, value_loss,
                 value_loss_ratio)


def run_training(model, nb_epochs, optimizer, policy_loss, train_loader, use_cuda, val_loader, value_loss,
                 value_loss_ratio):
    correct_cnt = 0
    total_cnt = 0
    sum_combined_loss = 0
    sum_policy_loss = 0
    sum_value_loss = 0
    writer = SummaryWriter()
    global_step = 0
    for epoch in range(nb_epochs):
        # training
        for batch_idx, (x, yv_train, yp_train) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            if use_cuda:
                x, yv_train, yp_train = x.cuda(), yv_train.cuda(), yp_train.cuda()
            x, yp_train = Variable(x), Variable(yp_train)
            value_out, policy_out = model(x)
            cur_policy_loss = policy_loss(policy_out, yp_train)
            cur_value_loss = value_loss(value_out, yv_train)

            _, pred_label = torch.max(policy_out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += float((pred_label == yp_train.data).sum())

            combined_loss = cur_policy_loss + value_loss_ratio * cur_value_loss
            sum_policy_loss += float(cur_policy_loss.data)
            sum_value_loss += float(cur_value_loss.data)
            sum_combined_loss += float(combined_loss.data)

            combined_loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                train_combined_loss = sum_combined_loss / 100
                train_value_loss = sum_value_loss / 100
                train_policy_loss = sum_policy_loss / 100
                train_policy_acc = correct_cnt / total_cnt

                val_combined_loss, val_value_loss, val_policy_loss, val_policy_acc = \
                    get_test_loss(model, value_loss_ratio, value_loss, policy_loss, use_cuda, val_loader)

                print(f'epoch: {epoch}, batch index: {batch_idx + 1}, train value loss: {train_value_loss},'
                      f' train policy loss: {train_policy_loss}, train policy acc: {train_policy_acc},'
                      f' train value loss: {train_value_loss}, val value loss: {val_value_loss},'
                      f' val policy loss: {val_policy_loss}, val policy acc: {val_policy_acc}')
                writer.add_scalars('Combined Loss', {'train': train_combined_loss,
                                                     'val': val_combined_loss}, global_step)
                writer.add_scalars('Value Loss', {'train': train_value_loss,
                                                  'val': val_value_loss}, global_step)
                writer.add_scalars('Policy Loss', {'train': train_policy_loss,
                                                   'val': val_policy_loss}, global_step)
                writer.add_scalars('Policy Accuracy', {'train': train_policy_acc,
                                                       'val': val_policy_acc}, global_step)
                sum_combined_loss = 0
                sum_policy_loss = 0
                sum_value_loss = 0

            global_step += 1


def get_test_loss(model, value_loss_ratio, value_loss, policy_loss, use_cuda, val_loader):
    correct_cnt = 0
    total_cnt = 0
    sum_combined_loss = 0
    sum_policy_loss = 0
    sum_value_loss = 0
    steps = 0
    for batch_idx, (x, yv_val, yp_val) in enumerate(val_loader):
        if use_cuda:
            x, yv_val, yp_val = x.cuda(), yv_val.cuda(), yp_val.cuda()
        x, yp_val = Variable(x, requires_grad=False), Variable(yp_val, requires_grad=False)
        value_out, policy_out = model(x)
        cur_value_loss = value_loss(value_out, yv_val)
        cur_policy_loss = policy_loss(policy_out, yp_val)

        _, pred_label = torch.max(policy_out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += float((pred_label == yp_val.data).sum())

        combined_loss = cur_policy_loss + value_loss_ratio * cur_value_loss
        sum_policy_loss += float(cur_policy_loss.data)
        sum_value_loss += float(cur_value_loss.data)
        sum_combined_loss += float(combined_loss.data)
        steps += 1

    return sum_combined_loss/steps, sum_value_loss/steps, sum_policy_loss/steps, correct_cnt/total_cnt


def prepare_dataset(f, test_size, batch_size, random_state):
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
