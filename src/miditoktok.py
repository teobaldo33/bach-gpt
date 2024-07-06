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
from transformers import AutoModelForCausalLM, MistralConfig, Trainer, TrainingArguments, GenerationConfig
from transformers.trainer_utils import set_seed
from tqdm import tqdm



# Seed
set_seed(777)

# Our tokenizer's configuration
BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
TOKENIZER_PARAMS = {
    "pitch_range": (0, 127),
    "beat_res": BEAT_RES,
    "num_velocities": 24,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": True,  # no multitrack here
    "num_tempos": 32,
    "tempo_range": (20, 100),  # (min_tempo, max_tempo)
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

# Creates the tokenizer
tokenizer = REMI(config)

# Trains the tokenizer with Byte Pair Encoding (BPE) to build the vocabulary, here 30k tokens
midi_paths = list(Path("bach-midi-collection").resolve().glob("**/*.mid")) + list(Path("bach-midi-collection").resolve().glob("**/*.midi"))
tokenizer.train(
    vocab_size=10000,
    files_paths=midi_paths,
)
tokenizer.save_params("tokenizer.json")
tokenizer.save_params("runs/remi_tokenizer.json")
tokenizer.save_pretrained("runs/tokenizer")


# Split MIDI paths in train/valid/test sets
total_num_files = len(midi_paths)
num_files_valid = round(total_num_files * 0.15)
num_files_test = round(total_num_files * 0.15)
shuffle(midi_paths)
midi_paths_valid = midi_paths[:num_files_valid]
midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
midi_paths_train = midi_paths[num_files_valid + num_files_test:]

# Chunk MIDIs and perform data augmentation on each subset independently
for files_paths, subset_name in (
        (midi_paths_train, "train"), (midi_paths_valid, "valid"), (midi_paths_test, "test")
):

    # Split the MIDIs into chunks of sizes approximately about 1024 tokens
    subset_chunks_dir = Path(f"bach_{subset_name}")
    split_files_for_training(
        files_paths=files_paths,
        tokenizer=tokenizer,
        save_dir=subset_chunks_dir,
        max_seq_len=256,
        num_overlap_bars=2,
    )

    # Perform data augmentation
    # augment_dataset(
    #     subset_chunks_dir,
    #     pitch_offsets=[-2, 2],
    #     velocity_offsets=[-1, 1],
    #     duration_offsets=[-0.5, 0.5],
    # )

# Create Dataset and Collator for training
midi_paths_train = list(Path("bach_train").glob("**/*.mid")) + list(Path("bach_train").glob("**/*.midi"))
midi_paths_valid = list(Path("bach_valid").glob("**/*.mid")) + list(Path("bach_valid").glob("**/*.midi"))
midi_paths_test = list(Path("bach_test").glob("**/*.mid")) + list(Path("bach_test").glob("**/*.midi"))
kwargs_dataset = {"max_seq_len": 256, "tokenizer": tokenizer, "bos_token_id": tokenizer["BOS_None"], "eos_token_id": tokenizer["EOS_None"]}
dataset_train = DatasetMIDI(midi_paths_train, **kwargs_dataset)
dataset_valid = DatasetMIDI(midi_paths_valid, **kwargs_dataset)
dataset_test = DatasetMIDI(midi_paths_test, **kwargs_dataset)

print("---------------------")
print(len(tokenizer))
      
# Creates model
model_config = MistralConfig(
    vocab_size=len(tokenizer),
    hidden_size=512,
    intermediate_size=2048,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=4,
    sliding_window=256,
    max_position_embeddings=8192,
    pad_token_id=tokenizer['PAD_None'],
    bos_token_id=tokenizer['BOS_None'],
    eos_token_id=tokenizer['EOS_None'],
)
model = AutoModelForCausalLM.from_config(model_config)



# Training

metrics = {metric: load_metric(metric) for metric in ["accuracy"]}

def compute_metrics(eval_pred):
    """
    Compute metrics for pretraining.

    Must use preprocess_logits function that converts logits to predictions (argmax or sampling).

    :param eval_pred: EvalPrediction containing predictions and labels
    :return: metrics
    """
    predictions, labels = eval_pred
    not_pad_mask = labels != -100
    labels, predictions = labels[not_pad_mask], predictions[not_pad_mask]
    return metrics["accuracy"].compute(predictions=predictions.flatten(), references=labels.flatten())

def preprocess_logits(logits: Tensor, _: Tensor) -> Tensor:
    """
    Preprocess the logits before accumulating them during evaluation.

    This allows to significantly reduce the memory usage and make the training tractable.
    """
    pred_ids = argmax(logits, dim=-1)  # long dtype
    return pred_ids

# Create config for the Trainer
USE_CUDA = cuda_available()
if not cuda_available():
    FP16 = FP16_EVAL = BF16 = BF16_EVAL = False
elif is_bf16_supported():
    BF16 = BF16_EVAL = True
    FP16 = FP16_EVAL = False
else:
    BF16 = BF16_EVAL = False
    FP16 = FP16_EVAL = True
USE_MPS = not USE_CUDA and mps_available()

training_config = TrainingArguments(
    "runs", False, True, True, False, "steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=3,
    eval_accumulation_steps=None,
    eval_steps=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_grad_norm=3.0,
    max_steps=1,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.3,
    log_level="debug",
    logging_strategy="steps",
    logging_steps=2,
    save_strategy="steps",
    save_steps=1,
    save_total_limit=5,
    use_cpu=True,
    seed=444,
    fp16=FP16,
    fp16_full_eval=FP16_EVAL,
    bf16=BF16,
    bf16_full_eval=BF16_EVAL,
    load_best_model_at_end=True,
    label_smoothing_factor=0.,
    optim="adamw_torch",
    report_to=["tensorboard"],
    gradient_checkpointing=True,
)

collator = DataCollator(tokenizer["PAD_None"], copy_inputs_as_labels=True)
trainer = Trainer(
    model=model,
    args=training_config,
    data_collator=collator,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    compute_metrics=compute_metrics,
    callbacks=None,
    preprocess_logits_for_metrics=preprocess_logits,
)

# Training
train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()
tokenizer.save_pretrained("runs/tokenizer")
tokenizer.save_params("runs/remi_tokenizer.json")



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
collator.pad_on_left = True
collator.eos_token = None
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