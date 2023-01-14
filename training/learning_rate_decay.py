import torch


class LearningRateDecayStrategy():
    def __init__(self, optimizer, epoch_delay, lr_decay):
        self.optimizer = optimizer
        self.epoch_delay = epoch_delay
        self.lr_decay = lr_decay

    def decay_learning_rate(self, prior_volume_passes, cur_volume_passes, complete_loss=0):
        pass

    @classmethod
    def create_instance(cls, args, optimizer):
        if args['smallify_decay'] == 0:
            return NeurcompDecayStrategy(optimizer, args['pass_decay'], args['lr_decay'])
        else:
            return SmallifyDecayStrategy(optimizer, args['smallify_decay'], args['lr_decay'], 1e-07)


# M: every pass_decay epochs reduce the lr by lr_decay
class NeurcompDecayStrategy(LearningRateDecayStrategy):
    def __init__(self, optimizer, pass_decay, lr_decay=0.2):
        super().__init__(optimizer, pass_decay, lr_decay)

    def decay_learning_rate(self, prior_volume_passes, cur_volume_passes, complete_loss=0):
        if prior_volume_passes != int(cur_volume_passes) and (int(cur_volume_passes) + 1) % self.epoch_delay == 0:
            print('------ learning rate decay ------', cur_volume_passes)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.lr_decay
        return False


# M: every smallify_decay epochs without improvement, reduce lr until threshold
class SmallifyDecayStrategy(LearningRateDecayStrategy):
    def __init__(self, optimizer, smallify_decay, lr_decay=0.1, lr_stop=1e-07):
        super().__init__(optimizer, smallify_decay, lr_decay)
        self.lr_stop = lr_stop
        self.last_loss = None
        self.no_gain_epoch = 0

    def decay_learning_rate(self, prior_volume_passes, cur_volume_passes, complete_loss=0):
        if prior_volume_passes != int(cur_volume_passes):
            if self.last_loss is None or complete_loss < self.last_loss:
                self.last_loss = complete_loss
                self.no_gain_epoch = 0
            else:
                self.no_gain_epoch += 1
            if self.no_gain_epoch == self.epoch_delay:
                for param_group in self.optimizer.param_groups:
                    if param_group['lr'] > self.lr_stop:
                        print('------ learning rate decay ------', cur_volume_passes)
                        param_group['lr'] *= self.lr_decay
                    else:
                        return True
                self.no_gain_epoch = 0
            return False
