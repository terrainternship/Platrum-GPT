import configparser
import logging

_config = None
__all__ = ['_config']


def get_config():
    global _config
    if not _config:
        _config = read_config('./config/default.config')
        local_config = read_config('./config/local.config')
        for section in local_config:
            if section not in _config:
                _config[section] = {}
            for key, value in local_config[section].items():
                _config[section][key] = value

    return _config


def read_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)

    cfg = {}
    for section in config.sections():
        cfg[section] = {}
        for key, value in config[section].items():
            cfg[section][key] = value

    return cfg


def configure_logging():
    get_config()
    logging_config = _config['Logging']
    level = logging_config.get('level')
    fmt = logging_config.get('format')
    filename = logging_config.get('file')
    logging.basicConfig(level=level, format=fmt, filename=filename)
