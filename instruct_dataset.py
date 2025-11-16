import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
import random
from transformers import WhisperFeatureExtractor
import os
import re

def format_answer_v1(raw: str, step: float = 0.2, max_tok: float = 30.0) -> str:
    def _repl(match: re.Match) -> str:
        num_str1 = match.group(1)
        num_str2 = match.group(2)

        def format_time(num_str: str) -> str:
            try:
                x = float(num_str)
                x = round(x, 1)
            except ValueError:
                return num_str                 
            x = round(round(x / step) * step, 1)
            if x == 0:
                return "<0.0>"
            tokens = []
            while x >= max_tok - 1e-4:  #
                tokens.append(f"<{max_tok:.1f}>")
                x = round(x - max_tok, 1)
            if x > 0:
                tokens.append(f"<{x:.1f}>")
            return "".join(tokens)

        return format_time(num_str1) + " - " + format_time(num_str2)

    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
    return pattern.sub(_repl, raw)

def format_answer_v2(raw: str) -> str:
    def _repl(match: re.Match) -> str:
        num_str1 = match.group(1)
        num_str2 = match.group(2)

        def format_time(num_str: str) -> str:
            try:
                x = float(num_str)
                x = round(x, 2)
            except ValueError:
                return num_str
            seq = []
            for ch in str(x):
                if ch == '.':
                    seq.append("<TIME_DOT>")
                else:
                    seq.append(f"<TIME_{ch}>")
            return "".join(seq)

        return format_time(num_str1) + " - " + format_time(num_str2)

    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
    return pattern.sub(_repl, raw)

def format_answer_v3(raw: str) -> str:
    def format_time_abs_offset(num_str: str) -> str:
        # Ensure the input is a valid number format, otherwise return it as is.
        try:
            x = float(num_str)
            rounded_x = round(x, 1)
            num_str = str(rounded_x)
        except ValueError:
            return num_str

        parts = num_str.split('.')
        integer_part = parts[0]
        fractional_part = parts[1] if len(parts) > 1 else None
        abs_tokens = [f"<a_{digit}>" for digit in integer_part]
        all_tokens = abs_tokens

        if fractional_part is not None:
            offset_tokens = [f"<f_{digit}>" for digit in fractional_part]
            all_tokens.extend(offset_tokens)

        return "".join(all_tokens)

    def _repl(match: re.Match) -> str:
        """The replacement function for re.sub"""
        num_str1 = match.group(1)
        num_str2 = match.group(2)
        return format_time_abs_offset(num_str1) + " - " + format_time_abs_offset(num_str2)
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
    return pattern.sub(_repl, raw)

class TimeAudioDataset(Dataset):
    def __init__(self, ann_path, whisper_path, format_tokens, max_len):
        super().__init__()
        data = []
        if isinstance(ann_path, str) and os.path.isdir(ann_path):
            json_files = [f for f in os.listdir(ann_path) if f.endswith(".json")]
            for json_file in json_files:
                file_path = os.path.join(ann_path, json_file)
                with open(file_path, "r") as f:
                    part_data = json.load(f)

                    part_data = random.sample(part_data, min(40000, len(part_data)))
                data.extend(part_data[:])
        elif os.path.isfile(ann_path):
            with open(ann_path, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Invalid path type: {ann_path}. It must be either a file or directory path.")
        
        self.annotation = []
        self.durations = []
        self.id_audios = []
        for item in data:
            self.annotation.append({
                "audio": item["audio"],
                "q": "<Speech><SpeechHere></Speech> "+item["question"],
                "a": item["answer"]
            })

            self.durations.append(item["duration"])
            self.id_audios.append(item["id"])
        print("Audio item number:", len(self.annotation))
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.format = format_tokens
        self.max_len = max_len
        print(f"Using {self.format}")

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        samples_spectrogram = [s["spectrogram"] for s in samples]
        max_num_chunks = max(s.size(0) for s in samples_spectrogram)
        if max_num_chunks==1:
            samples_spectrogram = [s.squeeze(0) for s in samples_spectrogram]
            cat_spectrogram = torch.stack(samples_spectrogram, dim=0) #(bz,80,3000)
        else:
            padded_spectrograms = []
            for spec in samples_spectrogram: #(3,80,3000)
                num_chunks, mel_bins, time_frames = spec.shape
                if num_chunks < max_num_chunks:
                    padding = torch.zeros((max_num_chunks - num_chunks, mel_bins, time_frames), dtype=spec.dtype, device=spec.device)
                    spec = torch.cat([spec, padding], dim=0)
                padded_spectrograms.append(spec)
            cat_spectrogram = torch.stack(padded_spectrograms, dim=0) #(bz,3,80,3000)

        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        padding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]
        duration = torch.tensor([(s["duration"]) for s in samples])
        id_audio = [s["id_audio"] for s in samples]

        return {
            "spectrogram": cat_spectrogram,
            "raw_wav": raw_wav,
            "padding_mask": padding_mask,
            "text": text,
            "task": task,
            "Q": Q,
            "id": id,
            "duration": duration,
            "id_audio": id_audio
        }

    def __getitem__(self, index):
        ann = self.annotation[index]
        audio_path = ann["audio"]
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            if len(audio.shape) == 2:  # stereo to mono
                audio = audio[:, 0]
            if len(audio) < sr:  # pad audio to at least 1s
                sil = np.zeros(sr - len(audio), dtype=float)
                audio = np.concatenate((audio, sil), axis=0)
            audio = audio[: sr * self.max_len]  # truncate audio to max_len            

            ## Process spectrogram
            spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
            if len(audio) <= sr * 30: #(1,80,3000)
                spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"] #.squeeze()
            else:
                spectrograms = []
                for start in range(0, len(audio), sr * 30):
                    end = min(start + sr * 30, len(audio))
                    audio_chunk = audio[start:end]
                    if len(audio_chunk) <= sr * 30 and len(audio_chunk)>=sr * 5:
                        audio_chunk = np.pad(audio_chunk, (0, sr * 30 - len(audio_chunk)), 'constant')
                    else:
                        continue
                    spectrogram = self.wav_processor(audio_chunk, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
                    spectrograms.append(spectrogram)  
                spectrogram = torch.stack(spectrograms, dim=0) #(3,80,3000)

            # text
            if self.format == "v1":
                text = format_answer_v1(ann["a"])
            elif self.format == "v2":
                text = format_answer_v2(ann["a"])
            elif self.format == "v3":
                text = format_answer_v3(ann["a"])  # answer
            else:
                text = ann["a"]
            task = ann.get("task", "asr")
            id_audio = self.id_audios[index]

            duration = (round(self.durations[index],1))
            question =  ann["q"]
            
            return {
                "spectrogram": spectrogram,
                "raw_wav": audio,
                "text": text,
                "task": task,
                "Q": question,
                "id": audio_path,
                "duration": duration,
                "id_audio": id_audio
            }
        except Exception as e:
            print(f"Failed to load examples with audio: {audio_path}. "
                  f"Will randomly sample an example as a replacement.")
            return self.__getitem__((index+1) % len(self.annotation))