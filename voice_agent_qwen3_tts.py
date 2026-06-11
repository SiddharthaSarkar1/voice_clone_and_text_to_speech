# -*- coding: utf-8 -*-
"""Voice Agent Qwen3-TTS.ipynb

!pip install -U qwen-tts
!git clone https://github.com/QwenLM/Qwen3-TTS.git

!cd Qwen3-TTS
!pip install -e Qwen3-TTS

!cd ./content/Qwen3-TTS
!pwd



import sys
import os

# Add the Qwen3-TTS directory to sys.path
repo_path = '/content/Qwen3-TTS'
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

print(f"sys.path updated: {repo_path in sys.path}")

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

"""# Custom Voice Generation"""

wavs, sr = model.generate_custom_voice(
    text="Hey all, the field of text-to-speech synthesis has seen remarkable advancements, allowing computers to generate highly natural and expressive human-like voices. These technologies are integral to many applications, including virtual assistants, audiobooks, and accessibility tools, continuously evolving to bridge the gap between artificial and human communication.",
    language="English",
    speaker="Vivian",
    instruct="Very happy."
)

sf.write("output_custom_voice.wav", wavs[0], sr)

wavs, sr = model.generate_custom_voice(
    text="Siddhartha, an AI engineer, is currently immersed in exploring Qwen3-TTS, a cutting-edge text-to-speech generation process. His objective is to thoroughly evaluate its capabilities, assess the quality of the generated voices, and understand the nuances of its performance across various linguistic and emotional parameters. This deep dive aims to uncover the potential and limitations of Qwen3-TTS in real-world applications, contributing to the advancement of AI-driven voice synthesis technology.",
    language="English",
    speaker="Ryan",
    instruct="Very happy."
)

sf.write("output_custom_voice_2.wav", wavs[0], sr)

"""# Voice Clone"""

from google.colab import files
uploaded = files.upload()

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

modelBase = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

ref_audio = 'voice_sample1.aac'
ref_text="Hey all, the field of text-to-speech synthesis has seen remarkable advancements, allowing computers to generate highly natural and expressive human-like voices. These technologies are integral to many applications, including virtual assistants, audiobooks, and accessibility tools, continuously evolving to bridge the gap between artificial and human communication."
speech="In the vast tapestry of human endeavor, the pursuit of knowledge stands as a beacon, guiding us through the complexities of existence. From the earliest stargazers who charted the cosmos with rudimentary tools, to the quantum physicists unraveling the universe's most intricate secrets, curiosity has been the driving force behind every significant advancement. This insatiable desire to understand, to innovate, and to connect, propels us forward, transforming not only our perception of the world but also our capacity to shape its future. Each discovery, whether grand or subtle, adds a thread to this fabric, enriching our collective understanding and opening new frontiers for exploration, reminding us that the journey of learning is endless and perpetually inspiring."


wavs, sr = modelBase.generate_voice_clone(
    text=speech,
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
    instruct="Very happy."
)

sf.write('voice_clone_sidd.wav', wavs[0], sr);

ref_audio = 'voice_sample1.aac'
ref_text="Hey all, the field of text-to-speech synthesis has seen remarkable advancements, allowing computers to generate highly natural and expressive human-like voices. These technologies are integral to many applications, including virtual assistants, audiobooks, and accessibility tools, continuously evolving to bridge the gap between artificial and human communication."

speech="Hey all, the field of text-to-speech synthesis has seen remarkable advancements, allowing computers to generate highly natural and expressive human-like voices. These technologies are integral to many applications, including virtual assistants, audiobooks, and accessibility tools, continuously evolving to bridge the gap between artificial and human communication."


wavs, sr = modelBase.generate_voice_clone(
    text=speech,
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
    instruct="Very happy."
)

sf.write('voice_clone_sidd2.wav', wavs[0], sr);