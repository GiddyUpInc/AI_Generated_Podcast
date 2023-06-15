# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets
# pip install datasets soundfile speechbrain
# pip install git+https://github.com/huggingface/transformers.git
# pip install --upgrade accelerate

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

#sf.write("speech new guy.wav", speech_newguy.numpy(), samplerate=16000)

class TTS:
    def __init__(self):
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
        # load xvector containing speaker's voice characteristics from a dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

        # Get Speaker embeddings
        self.speaker_a_embeddings = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)
        self.speaker_b_embeddings = torch.tensor(embeddings_dataset[7000]["xvector"]).unsqueeze(0)
        return

    def get_embeddings(self):
        return self.speaker_a_embeddings, self.speaker_b_embeddings

    def generate_speech(self, text: str, speaker_embedding):
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=self.vocoder)
        print("hello")
        return speech

    def save_audio(self, speech):
        sf.write("speech_new.wav", speech.numpy(), samplerate=16000)
        return