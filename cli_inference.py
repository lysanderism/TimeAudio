# 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
from transformers import WhisperFeatureExtractor

from config import Config
from models.timeaudio import TimeAudio
from utils import prepare_one_sample


parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=False, default="configs/infer_config.yaml", help='path to configuration file')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)

args = parser.parse_args()
cfg = Config(args)

model = TimeAudio.from_config(cfg.config.model)
model.to(args.device)
model.eval()

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

while True:
    try:
        print("=====================================")
        # wav_path = input("Your Wav Path:\n")
        # prompt = input("Your Prompt:\n")
        #******************  Dense Audio Caption  ****************************#
        wav_path  = "resource/Y2o1p83UjJFA.flac"
        prompt = "What are the sound events? Provide their time intervals and brief descriptions.\n"
        #******************  Temporal Audio Grounding  ****************************#
        # wav_path  = "resource/Y6NpPjovJ9j8.wav"
        # prompt = "What are the start and end times of audio matching 'dogs barking'?\n"
        samples = prepare_one_sample(wav_path, wav_processor)
        prompt = [
            cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
        ]
        print("Output:")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            print(model.generate(samples, cfg.config.generate, prompts=prompt)[0])
        # print(model.generate(samples, cfg.config.generate, prompts=prompt)[0])
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()