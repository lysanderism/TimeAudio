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

import logging
import json
import contextlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, StoppingCriteriaList, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from .Qformer import BertConfig, BertLMHeadModel
from .modeling_llama import LlamaForCausalLM, LlamaRotaryEmbedding
from .modeling_whisper import WhisperModel
from .beats.BEATs import BEATsConfig, BEATs
from .utils import StoppingCriteriaSub, decode_time_answer_v3, decode_time_answer_v2
import re

class TimeAudio(nn.Module):
    @classmethod
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("pretrained_model/bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        llama_path="",
        whisper_path="",
        freeze_whisper=True,
        beats_path="",
        freeze_beats=True,

        use_speech_Qformer=True,
        num_speech_query_token=1,
        freeze_speech_QFormer=False,
        window_level_Qformer=True,
        second_per_window=0.333333,
        second_stride=0.333333,
        
        speech_llama_proj_model="",
        freeze_speech_llama_proj=False,

        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,

        multi_prompt=False,
        prompt_path="",
        prompt_template="",
        max_txt_len=128,
        end_sym="</s>",
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        use_token_merge = False,
        max_time_pos=768,
    ):
        super().__init__()

        self.beats_path = beats_path
        self.use_speech_Qformer = use_speech_Qformer
        self.window_level_Qformer = window_level_Qformer
        self.second_per_window = second_per_window
        self.second_stride = second_stride
        self.lora = lora
        self.multi_prompt = multi_prompt
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.low_resource = low_resource

        logging.info('Loading LLaMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"

        logging.info('Loading LLaMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": device_8bit},
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLaMA Done')

        if self.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_model.print_trainable_parameters()
            logging.info('LoRA Training')

        assert whisper_path
        logging.info('Loading Whisper Model')
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
        if freeze_whisper:
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder.eval()
            logging.info("freeze Whisper")
        
        if self.beats_path:
            logging.info("Loading BEATs Model")
            beats_ckpt = torch.load(self.beats_path, map_location='cpu')
            beats_cfg = BEATsConfig(beats_ckpt['cfg'])
            self.beats = BEATs(beats_cfg)
            self.beats.load_state_dict(beats_ckpt['model'])
            self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
            if freeze_beats:
                for name, param in self.beats.named_parameters():
                    param.requires_grad = False
                self.beats.eval()
                logging.info("freeze BEATs")

        if self.use_speech_Qformer:
            if self.beats_path:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim
                )
            else:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model
                )
            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.speech_Qformer.cls = None
            if freeze_speech_QFormer:
                for name, param in self.speech_Qformer.named_parameters():
                    param.requires_grad = False
                self.speech_Qformer.eval()
                self.speech_query_tokens.requires_grad = False
                logging.info("freeze Speech QFormer")

            logging.info('Loading speech LLAMA proj')
            self.speech_llama_proj = nn.Linear(
                self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )
            if speech_llama_proj_model:
                logging.info("Loading speech LLAMA proj from {}".format(speech_llama_proj_model))
                speech_llama_proj_weight = torch.load(speech_llama_proj_model, map_location="cpu")
                self.load_state_dict(speech_llama_proj_weight['model'], strict=False)
            if freeze_speech_llama_proj:
                for name, param in self.speech_llama_proj.named_parameters():
                    param.requires_grad = False
                self.speech_llama_proj.eval()
                logging.info("freeze speech LLAMA proj")
        else:
            # feel free to add other aligners here
            raise NotImplementedError

        # prepare prompts
        self.prompt_dict = {}
        self.prompt_template = prompt_template
        if prompt_path:
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if "<SpeechHere>" in raw_prompt]
                self.prompt_dict[task] = [prompt_template.format(p) for p in filted_prompts]
            print("Loading training prompts done!")

        self.max_time_pos = max_time_pos
        if self.max_time_pos:
            self.abs_time_position_embedding = nn.Parameter(torch.zeros((max_time_pos, 2048), requires_grad=False).float())
        else:
            print('Random init real time!')
            self.abs_time_position_embedding = nn.Parameter(torch.randn((max_time_pos, 2048), requires_grad=True).float())

        self.use_token_merge  = use_token_merge
        if self.use_token_merge:
            self.dominant_token_num = 22
            self.contextual_token_num = 4
            def _checkpointed_forward(self, *inputs, **kwargs):
                return torch.utils.checkpoint.checkpoint(self._orig_forward, *inputs, **kwargs,use_reentrant=False)

            for layer in self.speech_Qformer.bert.encoder.layer:
                layer._orig_forward = layer.forward
                layer.forward = _checkpointed_forward.__get__(layer, type(layer))

    def _encode_auditory_feature(self, speech_embeds, audio_embeds=None, absolute_timestamp = []):
        with self.maybe_autocast():
            if self.use_speech_Qformer:
                speech_embeds = self.ln_speech(speech_embeds)
                if audio_embeds is not None:
                    audio_embeds = self.ln_audio(audio_embeds)
                    if audio_embeds.size(1) < speech_embeds.size(1):
                        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
                    elif audio_embeds.size(1) > speech_embeds.size(1):
                        speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
                    speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1) # ([bz, 1500, 1024+768])
                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device) # ([bz, 1500])
                if self.window_level_Qformer:
                    B, T, C = speech_embeds.shape
                    kernel = round(1500 * self.second_per_window / 30.0) ##17
                    stride = round(1500 * self.second_stride / 30.0) ##17
                    kernel = (1, kernel) # (1,17)
                    stride = (1, stride) # (1,17)
                    speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2) # ([bz, 2048, 1, 1500])
                    speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride) ## ([1, 34816, 88])
                    _, _, L = speech_embeds_overlap.shape ## L=88
                    speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L) # ([bz, 2048, 17, 88])
                    speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1]) # ([bz, 88, 17, 2048])
                    speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C) # ([88*bz, 17, 2048])
                    speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)
                if self.window_level_Qformer:
                    frame_times = torch.arange(1,L+1, device=speech_embeds.device) * self.second_stride
                    window_absolute_timestamps = frame_times.expand(B, L).clone()                  # (bz,L)
                    true_lengths = (absolute_timestamp / self.second_stride).floor().long()         # (bz,)

                    for b in range(B):
                        valid = true_lengths[b].item()
                        if valid < L:
                            window_absolute_timestamps[b, valid:] = 0.00  # zero out padded part
                    final_indices = torch.clamp(
                        window_absolute_timestamps.view(-1).div(self.second_stride).long(),
                        0,
                        self.max_time_pos - 1,
                    )  #  [B*L,]
                    time_position_embeddings = torch.matmul(
                        F.one_hot(final_indices, self.max_time_pos).float(), 
                        self.abs_time_position_embedding
                    )
                    # to [B*L, 1, 768]
                    time_position_embeddings = time_position_embeddings.unsqueeze(1)
                    time_position_embeddings = time_position_embeddings.expand(-1, kernel[1], -1)
                    speech_embeds += time_position_embeddings.to(speech_embeds.dtype)
                query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1) # ([88*bz, 1, 768])
                query_output = self.speech_Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=speech_embeds,
                    encoder_attention_mask=speech_atts,
                    return_dict=True,
                    output_attentions=self.use_token_merge
                )
                
                if self.use_token_merge and L>88:
                    attn = query_output.cross_attentions[-1] # cross_attentions: (B', H, Q, S)
                    attn = attn.view(B, L, attn.size(1), attn.size(2), attn.size(3))  # (B,L,H,Q,S)
                    attn = attn[:, :, :, 0, :].mean(2)                         # (B,L,S)
                    hidden = query_output.last_hidden_state.view(B, L, -1).contiguous()     # (B, L, D) speech_embeds # 
                    scores = attn.mean(dim=-1)# (B, L)
                    #dominant tokens
                    num_segments = int(2*L / 88)
                    segment_len = L // num_segments
                    merged_chunks = []
                    for i in range(num_segments):
                        start_idx = i * segment_len
                        end_idx = (i + 1) * segment_len if i < num_segments - 1 else L
                        hid = hidden[:, start_idx:end_idx, :]
                        score = scores[:, start_idx:end_idx]
                        current_chunk_len = hid.shape[1]
                        if current_chunk_len == 0:
                            continue
                        topk_idx = score.topk(self.dominant_token_num, dim=1).indices            # (B, K)
                        topk_idx_sorted, _ = topk_idx.sort(dim=1)          # keep time order
                        dominant_tok = hid.gather(1, topk_idx_sorted.unsqueeze(-1).expand(-1, -1, hid.size(-1)))

                        dom_mask = torch.zeros(B, hid.size(1), dtype=torch.bool, device=hid.device)
                        dom_mask.scatter_(1, topk_idx_sorted, True)
                        non_dom_mask = ~dom_mask
                        #contextual tokens
                        rem_hid = hid.masked_select(non_dom_mask.unsqueeze(-1)).view(B, -1, hid.size(-1))
                        if rem_hid.shape[1]:                         
                            metric  = F.normalize(rem_hid, p=2, dim=-1)                       # (B,R,D)
                            step = max(1, metric.size(1) // self.contextual_token_num)
                            target_indices = torch.arange(0, metric.size(1), step, device=hid.device)[:self.contextual_token_num]
                            tgt_tokens   = metric[:, target_indices, :]                                 # (B,Ctx,D)
                            tokens_to_merge = metric[:, ~torch.isin(torch.arange(metric.shape[1], device=metric.device),target_indices), :]  
                            
                            similarity = torch.bmm(tokens_to_merge, tgt_tokens.transpose(1,2))  # (B, R-C, C)
                            assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], self.contextual_token_num, dtype=rem_hid.dtype, device=metric.device)
                            assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
                            # assign_one_hot  = torch.zeros_like(similarity).scatter_(2, similarity.argmax(2, True), 1.)
                            counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)   # (B, Ctx, 1)
                            hidden_to_merge = rem_hid[:, ~torch.isin(torch.arange(rem_hid.shape[1], device=rem_hid.device),target_indices), :] 
                            aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
                            tgt_hid = rem_hid[:, target_indices,:]              # (B,C,D)
                            contextual_tok = tgt_hid + aggregated_hidden            # (B,C,D)
                        else:
                            contextual_tok = torch.empty(B, 0, hid.size(-1), device=hid.device)# R=0
                        merged_chunks.append(torch.cat([dominant_tok, contextual_tok], dim=1))
                    # speech_embeds = torch.cat([dominant_tok, contextual_tok], dim=1)  # (B, K+C, D)
                    speech_embeds = torch.cat(merged_chunks, dim=1)
                    speech_embeds = self.speech_llama_proj(speech_embeds)
                #project
                else:
                    speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)

                    if self.window_level_Qformer:
                        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
            else:
                raise NotImplementedError

        return speech_embeds, speech_atts

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None, absolute_timestamp = []):
        with self.maybe_autocast():
            if spectrogram.ndim == 3: #(bz, 80, 3000)
                speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state #(bz, 1500, 1024)
            else:
                bz = spectrogram.shape[0] #(bz, 3, 80, 3000)
                speech_embeds = []
                for i in range(spectrogram.shape[1]): # Process each chunk
                    speech_chunk = spectrogram[:,i] #(bz, 80, 3000)
                    speech_embed = self.speech_encoder(speech_chunk, return_dict=True).last_hidden_state
                    speech_embeds.append(speech_embed)
                speech_embeds = torch.stack(speech_embeds,dim=1).view(bz, -1, speech_embeds[0].shape[-1])#(bz, 1500*3, 1024)
            if self.beats_path and raw_wav is not None:
                audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
            else:
                audio_embeds = None

        return self._encode_auditory_feature(speech_embeds, audio_embeds=audio_embeds, absolute_timestamp = absolute_timestamp)

    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)
                
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False,padding="longest"
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids)

                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(embeds.device)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = embeds.shape[0]
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            return wrapped_embeds, wrapped_atts
        else:
            return embeds, atts

    def forward(self, samples, verbose=False):
        # detect whether there are multi tasks in this batch
        task = list(set(samples["task"]))
        if len(task) > 1:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:                
                if "Q" in samples:
                    prompt = [self.prompt_template.format(p) for p in samples["Q"]]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask, absolute_timestamp=samples["duration"])

        # wrap speech_embeds with prompts
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM
        text = [t + self.end_sym for t in samples["text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spectrogram.device)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] + 1],
                dtype=torch.long
            ).to(spectrogram.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)

        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

        if verbose:
            nvocab = self.llama_model.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
            mask = (labels != -100)
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])

        if verbose:
            return {"loss": loss, "correct": correct, "total": total}

        return {"loss": loss}

    def generate(self, samples, generate_cfg, prompts=None):
        batch_size = samples["spectrogram"].shape[0]

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask, absolute_timestamp=samples["duration"])

        if prompts is not None:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=True)

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)

        stop_words_ids = [torch.tensor([2]).cuda()]  
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
        )
        text = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False)
        # logging.info(f"{text}")

        if '<f_' in text[0]:
            text =  decode_time_answer_v3(text)
        # else:
        #     text = decode_time_answer_v2(text)
        return text

    @classmethod
    def from_config(cls, config):
        llama_path = config.get("llama_path")
        whisper_path = config.get("whisper_path")
        freeze_whisper = config.get("freeze_whisper", True)
        beats_path = config.get("beats_path", "")
        freeze_beats = config.get("freeze_beats", True)

        use_speech_Qformer = config.get("use_speech_Qformer", True)
        num_speech_query_token = config.get("num_speech_query_token", 1)
        freeze_speech_QFormer = config.get("freeze_speech_QFormer", False)
        window_level_Qformer = config.get("window_level_Qformer", True)
        second_per_window = config.get("second_per_window", 0.333333)
        second_stride = config.get("second_stride", 0.333333)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)

        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "</s>")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)
        added_time_token = config.get("added_time_token", "v3")
        tune_added_tokens = config.get("tune_added_tokens", "v3")
        use_token_merge = config.get("use_token_merge", False)

        model = cls(
            llama_path=llama_path,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            beats_path=beats_path,
            freeze_beats=freeze_beats,
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            window_level_Qformer=window_level_Qformer,
            second_per_window=second_per_window,
            second_stride=second_stride,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            use_token_merge=use_token_merge,
        )
        if added_time_token=="v1":
            SEGMENT_TOKEN_FORMAT = '<{:.1f}>'
            SPECIAL_TIME_TOKENS = [SEGMENT_TOKEN_FORMAT.format(round(i/10,1)) for i in range(0, 302, 2)]
            num_new_tokens = model.llama_tokenizer.add_tokens(SPECIAL_TIME_TOKENS, special_tokens=True)
            logging.info('Adding {} new tokens'.format((SPECIAL_TIME_TOKENS)))
        elif added_time_token=="v2":
            SPECIAL_TIME_TOKENS = ['<TIME_{}>'.format(i) for i in range(10)]
            SPECIAL_TIME_TOKENS.append('<TIME_DOT>')
            num_new_tokens = model.llama_tokenizer.add_tokens(SPECIAL_TIME_TOKENS)
            logging.info('Adding {} new tokens'.format((SPECIAL_TIME_TOKENS)))
        elif added_time_token=="v3":
            SPECIAL_TIME_TOKENS = ['<a_{}>'.format(i) for i in range(10)]
            OFFSET_TOKENS = ['<f_{}>'.format(i) for i in range(10)] # .xx
            SPECIAL_TIME_TOKENS.extend(OFFSET_TOKENS)
            num_new_tokens = model.llama_tokenizer.add_tokens(SPECIAL_TIME_TOKENS)
            logging.info('Adding {} new tokens'.format((SPECIAL_TIME_TOKENS)))
        else:
            pass
        model.llama_model.resize_token_embeddings(len(model.llama_tokenizer), mean_resizing=False)

        if tune_added_tokens=="v1":
            # v1 <0.2> tokens
            original_tokens = [token for token, idx in model.llama_tokenizer.get_vocab().items()]
            original_llama_ids  = model.llama_tokenizer.convert_tokens_to_ids(original_tokens)
            new_llama_ids  = model.llama_tokenizer.convert_tokens_to_ids(SPECIAL_TIME_TOKENS)
            embed_layer = model.llama_model.base_model.model.model.embed_tokens
            lm_head_layer = model.llama_model.base_model.model.lm_head
            with torch.no_grad():
                for token_str in SPECIAL_TIME_TOKENS:
                    new_token_id = model.llama_tokenizer.convert_tokens_to_ids(token_str)
                    # e.g., for "<2.4>", this gets the tokens for '<', '2', '.', '4', '>'
                    constituent_tokens = list(token_str)
                    constituent_ids = model.llama_tokenizer.convert_tokens_to_ids(constituent_tokens)
                    known_ids = [_id for _id in constituent_ids if _id != model.llama_tokenizer.unk_token_id]

                    if known_ids:
                        new_embedding = embed_layer.weight[known_ids].mean(dim=0)
                        new_lm_head_weight = lm_head_layer.weight[known_ids].mean(dim=0)
                    embed_layer.weight[new_token_id] = new_embedding
                    lm_head_layer.weight[new_token_id] = new_lm_head_weight
            if lora:
                embed_layer.weight.requires_grad = False
                lm_head_layer.weight.requires_grad=False
                embed_layer.weight[new_llama_ids].requires_grad = True
                lm_head_layer.weight[new_llama_ids].requires_grad=True
                logging.info(f"Frozen {embed_layer.weight.size(0)-len(new_llama_ids)} old embeddings;" f"tuning {len(new_llama_ids)} new embeddings")
        elif tune_added_tokens=="v2":
            # v2 tuned 0-9. tokens
            original_tokens = ['{}'.format(i) for i in range(10)]+['.']
            logging.info(original_tokens)
            original_llama_ids = [model.llama_tokenizer.encode(i)[-1] for i in original_tokens]
            new_llama_ids = [model.llama_tokenizer.encode(i)[-1] for i in SPECIAL_TIME_TOKENS]
            logging.info('initializing tokenizer weights')

            if lora:
                #time_token_initialization
                model.llama_model.base_model.model.model.embed_tokens.weight.data[new_llama_ids] = model.llama_model.base_model.model.model.embed_tokens.weight.data[original_llama_ids]
                model.llama_model.base_model.model.lm_head.weight.data[new_llama_ids] = model.llama_model.base_model.model.lm_head.weight.data[original_llama_ids]
                model.llama_model.base_model.model.model.embed_tokens = model.llama_model.base_model.model.model.embed_tokens.float()
                model.llama_model.base_model.model.model.embed_tokens.weight.requires_grad = True
                # model.llama_model.base_model.model.model.embed_tokens.weight[new_llama_ids].requires_grad = True
                logging.info(f"Frozen old embeddings;" f"tuning {len(new_llama_ids)} new embeddings")
                model.llama_model.base_model.model.lm_head = model.llama_model.base_model.model.lm_head.float()
                model.llama_model.base_model.model.lm_head.weight.requires_grad=True
                # model.llama_model.base_model.model.lm_head.weight[new_llama_ids].requires_grad=True
        elif tune_added_tokens=="v3":
            # v3 tuned anchor and offset tokens
            original_tokens = ['{}'.format(i) for i in range(10)]
            logging.info(original_tokens)
            original_llama_ids = [model.llama_tokenizer.encode(i)[-1] for i in original_tokens]
            new_llama_ids = [model.llama_tokenizer.encode(i)[-1] for i in SPECIAL_TIME_TOKENS]
            logging.info('initializing tokenizer weights')

            if lora:
                #time_token_initialization
                for t in SPECIAL_TIME_TOKENS:
                    new_id = model.llama_tokenizer.convert_tokens_to_ids(t)
                    digit = int(re.search(r"\d+", t).group())
                    if "a_" in t:                   
                        src_id = model.llama_tokenizer.convert_tokens_to_ids(str(digit))
                        src_vec = model.llama_model.base_model.model.model.embed_tokens.weight.data[src_id].clone()
                    else:                              
                        dot_id   = model.llama_tokenizer.convert_tokens_to_ids(".")
                        digit_id = model.llama_tokenizer.convert_tokens_to_ids(str(digit))
                        src_vec = model.llama_model.base_model.model.model.embed_tokens.weight.data[[dot_id, digit_id]].mean(0)
                        # src_vec += 0.01 * torch.randn_like(src_vec)
                    model.llama_model.base_model.model.model.embed_tokens.weight.data[new_id]=src_vec
                    model.llama_model.base_model.model.lm_head.weight.data[new_id]=src_vec

                model.llama_model.base_model.model.model.embed_tokens = model.llama_model.base_model.model.model.embed_tokens.float()
                model.llama_model.base_model.model.model.embed_tokens.weight.requires_grad = True
                # model.llama_model.base_model.model.model.embed_tokens.weight[new_llama_ids].requires_grad = True
                model.llama_model.base_model.model.lm_head = model.llama_model.base_model.model.lm_head.float()
                model.llama_model.base_model.model.lm_head.weight.requires_grad=True
                # model.llama_model.base_model.model.lm_head.weight[new_llama_ids].requires_grad=True
                logging.info(f"tuning {len(new_llama_ids)} new embeddings")
        else:
            if lora:
                model.llama_model.base_model.model.model.embed_tokens.weight.requires_grad = False
                model.llama_model.base_model.model.lm_head.weight.requires_grad = False

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            for k in ckpt["model"].keys():
                print((f"[KEY] {k}"))
            model.load_state_dict(ckpt['model'], strict=False)

        return model