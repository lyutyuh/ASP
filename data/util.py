from os import makedirs
from os.path import join
import numpy as np
import pyhocon
import logging
import torch
import random

logger = logging.getLogger(__name__)


def flatten(l):
    return [item for sublist in l for item in sublist]
