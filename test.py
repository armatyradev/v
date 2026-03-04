import pytest
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration


MODEL_NAME = "facebook/musicgen-small"
# MODEL_NAME = "./model/"


@pytest.fixture(scope="module")
def model_and_processor():
    """Загрузки модели и процессора для тестов."""
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
    return model, processor


def test_model_and_processor_loading(model_and_processor):
    """
    Тест 1: Проверяет, что модель и процессор успешно загружаются
    и являются корректными объектами.
    """
    model, processor = model_and_processor
    assert model is not None
    assert processor is not None
    assert isinstance(model, MusicgenForConditionalGeneration)


def test_input_processing(model_and_processor):
    """
    Тест 2: Проверяет, что процессор правильно обрабатывает входной текст
    и возвращает тензоры с ключом 'input_ids'.
    """
    _, processor = model_and_processor
    inputs = processor(
        text=["80s pop track with bassy drums and synth"],
        padding=True,
        return_tensors="pt"
    )
    assert "input_ids" in inputs
    assert inputs["input_ids"].shape[0] == 1 
    assert isinstance(inputs["input_ids"], torch.Tensor)


def test_audio_generation(model_and_processor):
    """
    Тест 3: Проверяет, что модель генерирует аудио-тензор корректной формы:
    (batch, channels, samples), и количество сэмплов больше 0.
    """
    model, processor = model_and_processor
    inputs = processor(
        text=["test track"],
        padding=True,
        return_tensors="pt"
    )
    audio_values = model.generate(**inputs, max_new_tokens=128)
    assert audio_values is not None
    assert isinstance(audio_values, torch.Tensor)
    assert len(audio_values.shape) == 3 
    assert audio_values.shape[1] == 1 
    assert audio_values.shape[2] > 0  


def test_sampling_rate_and_dtype(model_and_processor):
    """
    Тест 4: Проверяет, что частота дискретизации задана корректно (int > 0)
    и что сгенерированные данные имеют правильный тип (float32 или float).
    """
    model, processor = model_and_processor
    sampling_rate = model.config.audio_encoder.sampling_rate
    assert isinstance(sampling_rate, int)
    assert sampling_rate > 0

    inputs = processor(
        text=["short test"],
        padding=True,
        return_tensors="pt"
    )
    audio_values = model.generate(**inputs, max_new_tokens=16)
    assert isinstance(audio_values, torch.Tensor)
    assert audio_values.dtype in [torch.float32, torch.float]
