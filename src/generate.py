from copy import deepcopy
from pathlib import Path
from random import shuffle

import torch
from evaluate import load as load_metric
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.pytorch_data import split_files_for_training
from miditok.data_augmentation import augment_dataset
from torch import Tensor, argmax
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_available, is_bf16_supported
from torch.backends.mps import is_available as mps_available
from transformers import AutoModelForCausalLM, MistralConfig, Trainer, TrainingArguments, GenerationConfig, AutoTokenizer
from transformers.trainer_utils import set_seed
from tqdm import tqdm
import json
import json
from pathlib import Path

import os

model_path = Path("./runs/")
config_path = Path("./tokenizer.json")
with config_path.open() as f:
    config = json.load(f)
# Create TokenizerConfig object directly from the loaded JSON
tokenizer_config = TokenizerConfig(**config['config'])
# Create the REMI tokenizer
tokenizer = REMI(tokenizer_config)
print(f"Tokenizer vocabulary size: {len(tokenizer)}")
# Load the model
model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
# Generation

(gen_results_path := Path('gen_res')).mkdir(parents=True, exist_ok=True)
generation_config = GenerationConfig(
    max_new_tokens=200,  # extends samples by 200 tokens
    num_beams=1,         # no beam search
    do_sample=True,      # but sample instead
    temperature=0.9,
    top_k=15,
    top_p=0.95,
    epsilon_cutoff=3e-4,
    eta_cutoff=1e-3,
    pad_token_id=tokenizer.pad_token_id,
)

# Here the sequences are padded to the left, so that the last token along the time dimension
# is always the last token of each seq, allowing to efficiently generate by batch
collator = DataCollator(tokenizer["PAD_None"], copy_inputs_as_labels=True)
collator.pad_on_left = True
collator.eos_token = None
midi_paths_test = list(Path("bach_test").glob("**/*.mid")) + list(Path("bach_test").glob("**/*.midi"))
kwargs_dataset = {"max_seq_len": 256, "tokenizer": tokenizer, "bos_token_id": tokenizer["BOS_None"], "eos_token_id": tokenizer["EOS_None"]}
dataset_test = DatasetMIDI(midi_paths_test, **kwargs_dataset)
dataloader_test = DataLoader(dataset_test, batch_size=16, collate_fn=collator)
model.eval()
count = 0
for batch in tqdm(dataloader_test, desc='Testing model / Generating results'):  # (N,T)
    res = model.generate(
        inputs=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
        generation_config=generation_config)  # (N,T)

    # Saves the generated music, as MIDI files and tokens (json)
    for prompt, continuation in zip(batch["input_ids"], res):
        generated = continuation[len(prompt):]
        print(generated)
        midi = tokenizer.decode(deepcopy(generated))
        tokens = [generated, prompt, continuation]  # list compr. as seqs of dif. lengths
        tokens = [seq.tolist() for seq in tokens]
        for tok_seq in tokens[1:]:
            _midi = tokenizer.decode(deepcopy(tok_seq))
            midi.tracks.append(_midi.tracks[0])
        midi.tracks[0].name = f'Continuation of original sample ({len(generated)} tokens)'
        midi.tracks[1].name = f'Original sample ({len(prompt)} tokens)'
        midi.tracks[2].name = f'Original sample and continuation'
        midi.dump_midi(gen_results_path / f'{count}.mid')
        tokenizer.save_tokens(tokens, gen_results_path / f'{count}.json')

        count += 1