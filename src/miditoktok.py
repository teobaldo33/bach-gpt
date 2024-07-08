import argparse
from copy import deepcopy
from pathlib import Path
from random import shuffle

import torch
from evaluate import load as load_metric
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator, split_files_for_training
from torch import Tensor, argmax
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_available, is_bf16_supported
from transformers import AutoModelForCausalLM, MistralConfig, Trainer, TrainingArguments, GenerationConfig
from transformers.trainer_utils import set_seed
from tqdm import tqdm

def setup_tokenizer(midi_paths):
    config = TokenizerConfig(
        pitch_range=(0, 127),
        beat_res={(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1},
        num_velocities=24,
        special_tokens=["PAD", "BOS", "EOS"],
        use_chords=True, use_rests=True, use_tempos=True,
        use_time_signatures=True, use_programs=True,
        num_tempos=32, tempo_range=(20, 100),
    )
    tokenizer = REMI(config)
    tokenizer.train(vocab_size=35000, files_paths=midi_paths)
    return tokenizer

def split_and_process_dataset(midi_paths, tokenizer):
    total_num_files = len(midi_paths)
    num_files_valid = round(total_num_files * 0.15)
    num_files_test = round(total_num_files * 0.15)
    shuffle(midi_paths)
    midi_paths_valid = midi_paths[:num_files_valid]
    midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
    midi_paths_train = midi_paths[num_files_valid + num_files_test:]

    for files_paths, subset_name in ((midi_paths_train, "train"), (midi_paths_valid, "valid"), (midi_paths_test, "test")):
        subset_chunks_dir = Path(f"bach_{subset_name}")
        split_files_for_training(files_paths, tokenizer, subset_chunks_dir, max_seq_len=256, num_overlap_bars=2)

    return (list(Path(f"bach_{subset}").glob("**/*.mid")) + list(Path(f"bach_{subset}").glob("**/*.midi"))
            for subset in ["train", "valid", "test"])

def setup_datasets(midi_paths_train, midi_paths_valid, midi_paths_test, tokenizer):
    kwargs_dataset = {"max_seq_len": 256, "tokenizer": tokenizer,
                      "bos_token_id": tokenizer["BOS_None"], "eos_token_id": tokenizer["EOS_None"]}
    return (DatasetMIDI(paths, **kwargs_dataset)
            for paths in [midi_paths_train, midi_paths_valid, midi_paths_test])

def setup_model(tokenizer, device):
    model_config = MistralConfig(
        vocab_size=len(tokenizer), hidden_size=512, intermediate_size=2048,
        num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
        sliding_window=256, max_position_embeddings=8192,
        pad_token_id=tokenizer['PAD_None'],
        bos_token_id=tokenizer['BOS_None'],
        eos_token_id=tokenizer['EOS_None'],
    )
    return AutoModelForCausalLM.from_config(model_config).to(device)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    not_pad_mask = labels != -100
    labels, predictions = labels[not_pad_mask], predictions[not_pad_mask]
    return load_metric("accuracy").compute(predictions=predictions.flatten(), references=labels.flatten())

def preprocess_logits(logits: Tensor, _: Tensor) -> Tensor:
    return argmax(logits, dim=-1)

def setup_trainer(model, training_config, collator, dataset_train, dataset_valid):
    return Trainer(
        model=model,
        args=training_config,
        data_collator=collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        compute_metrics=compute_metrics,
        callbacks=None,
        preprocess_logits_for_metrics=preprocess_logits,
    )

def generate_results(model, tokenizer, dataset_test, device):
    gen_results_path = Path('gen_res')
    gen_results_path.mkdir(parents=True, exist_ok=True)
    generation_config = GenerationConfig(
        max_new_tokens=200, num_beams=1, do_sample=True,
        temperature=0.9, top_k=15, top_p=0.95,
        epsilon_cutoff=3e-4, eta_cutoff=1e-3,
        pad_token_id=tokenizer.pad_token_id,
    )

    collator = DataCollator(tokenizer["PAD_None"], copy_inputs_as_labels=True)
    collator.pad_on_left = True
    collator.eos_token = None
    dataloader_test = DataLoader(dataset_test, batch_size=16, collate_fn=collator)
    
    model.eval()
    count = 0
    for batch in tqdm(dataloader_test, desc='Testing model / Generating results'):
        res = model.generate(
            inputs=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            generation_config=generation_config)

        for prompt, continuation in zip(batch["input_ids"], res):
            generated = continuation[len(prompt):].cpu()
            midi = tokenizer.decode(deepcopy(generated))
            tokens = [generated, prompt, continuation]
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

def main(dataset_path):
    set_seed(777)
    device = torch.device("cuda" if cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    midi_paths = list(Path(dataset_path).glob("**/*.mid")) + list(Path(dataset_path).glob("**/*.midi"))
    tokenizer = setup_tokenizer(midi_paths)
    midi_paths_train, midi_paths_valid, midi_paths_test = split_and_process_dataset(midi_paths, tokenizer)
    dataset_train, dataset_valid, dataset_test = setup_datasets(midi_paths_train, midi_paths_valid, midi_paths_test, tokenizer)

    model = setup_model(tokenizer, device)

    bf16_supported = is_bf16_supported()
    training_config = TrainingArguments(
        "runs", False, True, True, False, "steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=3,
        eval_accumulation_steps=None,
        eval_steps=200,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=3.0,
        max_steps=20000,
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=0.3,
        log_level="debug",
        logging_strategy="steps",
        logging_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,
        seed=444,
        fp16=not bf16_supported,
        fp16_full_eval=not bf16_supported,
        bf16=bf16_supported,
        bf16_full_eval=bf16_supported,
        load_best_model_at_end=True,
        label_smoothing_factor=0.,
        optim="adamw_torch",
        report_to=["tensorboard"],
        gradient_checkpointing=True,
        use_cpu=False,
    )

    collator = DataCollator(tokenizer["PAD_None"], copy_inputs_as_labels=True)
    trainer = setup_trainer(model, training_config, collator, dataset_train, dataset_valid)

    print("Starting training...")
    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    generate_results(model, tokenizer, dataset_test, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and generate MIDI using a custom model.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to directory containing MIDI files for training")
    args = parser.parse_args()
    main(args.dataset)
