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

import torch
from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
import re
from typing import List, Union
def decode_time_answer_v1(raw: str) -> str:
    TOKENS_SEQ = re.compile(r'(?:<\d+\.\d>)+')

    def _decode(match: re.Match) -> str:
        token_group = match.group(0)
        values = re.findall(r'<(\d+\.\d)>', token_group)
        total  = sum(float(v) for v in values)
        return f"{total:.1f}"

    return [TOKENS_SEQ.sub(_decode, s) for s in raw]

def decode_time_answer_v2(raw: Union[str, List[str]]) -> Union[str, List[str]]:
    def _repl(match: re.Match) -> str:
        token = match.group(0)
        if token == "<TIME_DOT>":
            return "."
        elif token.startswith("<TIME_") and token.endswith(">"):
            return token[6:-1]
        return token
    def _clean(s: str) -> str:
        out = pattern.sub(_repl, s)
        out = re.sub(r"\s{2,}", " ", out)                
        out = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", out)  # 0 . 4  -> 0.4
        out = re.sub(r"(?<=\d)\s+(?=\d)", "", out)        # 1 0    -> 10
        out = re.sub(r"\s*,\s*", ", ", out)               
        out = re.sub(r"\s*-\s*", " - ", out)              
        out = re.sub(r"\s+\.", ".", out)                 
        return out.strip()

    pattern = re.compile(r"<TIME_\d+>|<TIME_DOT>")

    if isinstance(raw, str):
        return _clean(raw) 

    return [_clean(s) for s in raw]#[pattern.sub(_repl, s) for s in raw]

def decode_time_answer_v3(raw):
    TOKEN = r"<(?:a|f)_\d+>"
    TOKEN_SEQ = re.compile(f"({TOKEN}(?:\\s*{TOKEN})*)")
    HYPHEN_SPACING = re.compile(r"\s+-\s+")
    COMMA_SPACING = re.compile(r"\s+,\s*")

    def _decode(m):
        seq = m.group(0)
        t = "".join(re.findall(r"<a_(\d+)>", seq)) or "0"
        f = "".join(re.findall(r"<f_(\d+)>", seq)) or "0"
        return f"{float(f'{t}.{f}'): .1f}".strip()

    def _process(s):
        s = TOKEN_SEQ.sub(_decode, s)
        s = HYPHEN_SPACING.sub(" - ", s)
        s = COMMA_SPACING.sub(", ", s)
        return s

    if isinstance(raw, str):
        return _process(raw)
    return [_process(s) for s in raw]