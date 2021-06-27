from typing import Optional

import torch
from torch.nn.functional import kl_div, log_softmax

from nn.PommerModel import PommerModel
from training.loss.masked_kl_div import MaskedKLDivergenceLoss


class Metrics:
    """
    Class which stores metric attributes as a struct
    """
    def __init__(self):
        self.correct_cnt = 0
        self.total_cnt = 0
        self.sum_combined_loss = 0
        self.sum_policy_loss = 0
        self.sum_policy_kl = 0
        self.sum_value_loss = 0
        self.steps = 0
        self.kl = MaskedKLDivergenceLoss()

    def reset(self):
        self.correct_cnt = 0
        self.total_cnt = 0
        self.sum_combined_loss = 0
        self.sum_policy_loss = 0
        self.sum_policy_kl = 0
        self.sum_value_loss = 0
        self.steps = 0

    def update(self, model: PommerModel, policy_loss, value_loss, value_loss_ratio, x_train, yv_train, ya_train, yp_train
               , model_state=None, ids=None, device=None):
        """
        Updates the metrics and calculates the combined loss

        :param model: Model to optimize
        :param policy_loss: Policy loss object
        :param value_loss: Value loss object
        :param value_loss_ratio: Value loss ratio
        :param x_train: Training samples
        :param yv_train: Target labels for value
        :param ya_train: Target labels for policy
        :param yp_train: Target policy distribution
        :param model_state: The current state of the model (optional)
        :param ids: The episode ids in the data (required for masking)
        :param device: The device (required for masking)
        :return: A tuple of the combined loss and (optionally) the next state
        """
        mask = None
        if ids is not None:
            mask = (ids != -1).to(dtype=torch.float).unsqueeze(-1)
            if device is not None:
                mask = mask.to(device=device)

        if len(x_train.shape) == 4:
            # regular batch
            nb_samples = x_train.shape[0]
        elif len(x_train.shape) == 5:
            # sequence
            if mask is None:
                nb_samples = x_train.shape[0] * x_train.shape[1]
            else:
                nb_samples = torch.sum(mask)
        else:
            raise ValueError(f"Unsupported input dimension! Got shape: {x_train.shape}")

        if model.is_stateful:
            value_out, policy_out, next_state = model(model.flatten(x_train, model_state))
        else:
            value_out, policy_out = model(model.flatten(x_train, None))

        # get the combined loss
        cur_policy_loss = policy_loss(policy_out, yp_train, mask=mask)
        cur_value_loss = value_loss(value_out, yv_train.unsqueeze(-1), mask=mask)
        combined_loss = (1 - value_loss_ratio) * cur_policy_loss + value_loss_ratio * cur_value_loss

        # update metrics
        self.total_cnt += nb_samples
        _, pred_label = torch.max(policy_out.data, -1)
        if mask is None:
            self.correct_cnt += float((pred_label == ya_train.data).sum())
        else:
            self.correct_cnt += float(((pred_label == ya_train.data) * mask.squeeze(-1)).sum())

        self.sum_policy_loss += float(cur_policy_loss.data)
        self.sum_policy_kl += float(self.kl(policy_out, yp_train, mask=mask))
        self.sum_value_loss += float(cur_value_loss.data)
        self.sum_combined_loss += float(combined_loss.data)

        self.steps += 1

        if model.is_stateful:
            return combined_loss, next_state
        else:
            return combined_loss, None

    def combined_loss(self):
        return self.sum_combined_loss / self.steps

    def value_loss(self):
        return self.sum_value_loss / self.steps

    def policy_loss(self):
        return self.sum_policy_loss / self.steps

    def policy_kl(self):
        return self.sum_policy_kl / self.steps

    def policy_acc(self):
        return self.correct_cnt / self.total_cnt

    def log_to_tensorboard(self, writer, global_step) -> None:
        """
        Logs all metrics to Tensorboard at the current global step.

        :param global_step: Global step of batch update
        :param metrics: Metrics which shall be logged
        :param writer: Tensorobard writer handle
        :return:
        """

        writer.add_scalar('Loss/Value', self.value_loss(), global_step)
        writer.add_scalar('Loss/Policy', self.policy_loss(), global_step)
        writer.add_scalar('Loss/Combined', self.combined_loss(), global_step)

        writer.add_scalar('Policy/Accuracy', self.policy_acc(), global_step)
        writer.add_scalar('Policy/KL divergence', self.policy_kl(), global_step)

        writer.flush()
