from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torch

import sys
sys.path.append("D:\\BIM-LM\\LM\\t5")
from utils.utils import Utils
from utils.data_processor import DataProcessor
from utils.data_utils import InputExample, InputFeature, load_examples

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
    "automatic-speech-recognition",
    model="../../../pretrained_models/whisper-base.en", # change this to the path of the model
    chunk_length_s=30,
    device=device,
    )

    sample = "./speech_input.wav"
    prediction = pipe(sample)["text"]

    return prediction

if __name__ == "__main__":
    prediction = main()
    print(prediction)
