import pytest
import torch
from diffusers import StableDiffusionPipeline
import numpy as np

MODEL_NAME = "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator"

@pytest.fixture(scope="module")
def model():
    model = StableDiffusionPipeline.from_pretrained(MODEL_NAME, device_map = None, low_cpu_mem_usage=True)
    return model

def test_model_and_processor_loading(model):
    model = model
    assert model is not None
    assert isinstance(model, StableDiffusionPipeline)
    assert model.text_encoder is not None
    assert model.unet is not None
    assert model.tokenizer is not None

def test_inference_and_image_creation(model):
    prompt = "magical human, pixel art, white background"
    output = model(
        prompt
    ).images[0]

    assert output is not None
    assert hasattr(output, "size")
    img_array = np.array(output)
    assert img_array.std() > 0