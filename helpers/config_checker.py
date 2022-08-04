import os


DEFAULT_CONFIG_DIR = 'configs'


class ConfigsRepo(object):
    def __init__(self, config_dir=DEFAULT_CONFIG_DIR):
        self.config_dir = config_dir

    def check_configs_available(self, *config_ps):
        for p in config_ps:
            assert self.config_dir in p, 'Expected {} to contain {}!'.format(p, self.config_dir)
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
