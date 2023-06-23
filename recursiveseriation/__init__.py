# -*- coding: utf-8 -*-
"""Top level package for mozzarella."""

from recursiveseriation.logger import configure_logging
from recursiveseriation.settings import init_settings

SETTINGS = init_settings()

logger = configure_logging("recursiveseriation", SETTINGS, kidnap_loggers=True)

__app_name__ = "recursiveseriation"
__version__ = "0.1.1"
