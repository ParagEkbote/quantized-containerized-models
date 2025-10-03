from io import BytesIO
from pathlib import Path

import pytest
import requests
import torch
from PIL import Image
from pytest import raises

from models.flux_fast_lora_hotswap.predict import (
    Predictor,
    load_image,
    login_with_env_token,
    save_image,
)
