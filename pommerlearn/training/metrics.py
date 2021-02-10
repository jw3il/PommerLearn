import torch
from torch.nn.functional import kl_div, log_softmax


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

    def reset(self):
        self.correct_cnt = 0
        self.total_cnt = 0
        self.sum_combined_loss = 0
        self.sum_policy_loss = 0
        self.sum_policy_kl = 0
        self.sum_value_loss = 0
        self.steps = 0

    def update(self, model, policy_loss, value_loss, value_loss_ratio, x_train, yv_train, ya_train, yp_train,
                       fit_pol_dist):
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
        :param fit_pol_dist: Whether to use the policy distribution for the policy loss target
        :return: The combined loss
        """

        # get the combined loss
        value_out, policy_out = model(x_train)
        # TODO: Improve code design, this should be handled automatically
        if fit_pol_dist:
            cur_policy_loss = policy_loss(policy_out, yp_train)
        else:
            cur_policy_loss = policy_loss(policy_out, ya_train)

        cur_value_loss = value_loss(value_out.view(-1), yv_train)
        combined_loss = (1 - value_loss_ratio) * cur_policy_loss + value_loss_ratio * cur_value_loss

        # update metrics
        _, pred_label = torch.max(policy_out.data, 1)
        self.total_cnt += x_train.data.size()[0]
        self.correct_cnt += float((pred_label == ya_train.data).sum())
        self.sum_policy_loss += float(cur_policy_loss.data)
        kl_value = kl_div(log_softmax(policy_out, dim=1), yp_train, reduction='batchmean')
        self.sum_policy_kl += float(kl_value)
        self.sum_value_loss += float(cur_value_loss.data)
        self.sum_combined_loss += float(combined_loss.data)
        self.steps += 1

        return combined_loss

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