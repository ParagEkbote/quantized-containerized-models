from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image

import pytest
from models.flux_fast_lora_hotswap.predict import (
    Predictor,
    load_image,
    login_with_env_token,
    save_image,
)
from pytest import raises