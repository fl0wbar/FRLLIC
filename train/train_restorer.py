import os

from helpers import logdir_helpers
from helpers.paths import get_ckpts_dir, get_experiment_dir, CKPTS_DIR_NAME
from helpers.saver import Restorer


class TrainRestorer(Restorer):
    @staticmethod
    def from_flags(restore_dir, log_dir, restore_continue, restore_itr,
                   restart_at_zero=False, strict='y'):
        if restore_dir is None:
            return None
        strict = {'y': True, 'n': False}[strict]
        return TrainRestorer(get_ckpts_dir(get_experiment_dir(log_dir, restore_dir)),
                             restore_continue, restore_itr, restart_at_zero, strict)

    def __init__(self, out_dir, restore_continue=False, restore_itr=-1, restart_at_zero=False, strict=True,
                 ckpt_name_fmt='ckpt_{:010d}.pt', tmp_postfix='.tmp'):
        """
        :param out_dir: ends in ckpts/
        :param restore_continue:
        :param restart_at_zero:
        :param ckpt_name_fmt:
        :param tmp_postfix:
        """
        assert out_dir.rstrip(os.path.sep).endswith(CKPTS_DIR_NAME), out_dir
        super(TrainRestorer, self).__init__(out_dir, ckpt_name_fmt, tmp_postfix)
        self.restore_continue = restore_continue
        self.restore_itr = restore_itr
        self.restart_at_zero = restart_at_zero
        self.strict = strict

    def restore_desired_ckpt(self, modules):
        itrc, ckpt_p = self.get_ckpt_for_itr(self.restore_itr)
        print('Restoring {}...'.format(itrc))
        return self.restore(modules, ckpt_p, self.strict)

    def get_log_dir(self):
        log_dir = os.path.dirname(self._out_dir)  # should be .../logs/MMDD_HHdd config config config
        assert logdir_helpers.is_log_dir(log_dir)
        return log_dir
