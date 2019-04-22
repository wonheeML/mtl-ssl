""" Common utilities. """

# Logging
# =======

import logging
import os, os.path
from colorlog import ColoredFormatter

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
#    datefmt='%H:%M:%S.%f',
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('small')
log.setLevel(logging.DEBUG)
log.handlers = []       # No duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO + 1, 'INFOV')
def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)
logging.Logger.infov = _infov


# Etc
# ===

def get_tempdir():
    import getpass, tempfile
    user = getpass.getuser()

    for t in ('/data1/' + user,
              '/data/' + user,
              tempfile.gettempdir()):
        if os.path.exists(t):
            return mkdir_p(t + '/small.tmp')
    return None


def get_specific_dir(name):
    import getpass

    assert name is not None, 'Need to specify directory name.'
    user = getpass.getuser()

    for t in ('/data2/' + user,
              '/data1/' + user,
              '/data/' + user):
        if os.path.exists(t):
            return mkdir_p(t + '/' + name)
    return None

def mkdir_p(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


__all__ = (
    'log', 'get_tempdir', 'get_specific_dir', 'mkdir_p',
)
