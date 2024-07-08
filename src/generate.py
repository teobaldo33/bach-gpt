import argparse
from copy import deepcopy
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import miditok
from miditok.pytorch_data import DatasetMIDI, DataCollator

def load_model_and_tokenizer(model_path, tokenizer_path):
    tokenizer = miditok.REMI.from_pretrained(Path(tokenizer_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
    return model, tokenizer

def setup_generation_config(tokenizer):
    return GenerationConfig(
        max_new_tokens=200, num_beams=1, do_sample=True,
        temperature=0.9, top_k=15, top_p=0.95,
        epsilon_cutoff=3e-4, eta_cutoff=1e-3,
        pad_token_id=tokenizer.pad_token_id,
    )

def setup_data_loader(midi_paths, tokenizer, batch_size=16):
    collator = DataCollator(tokenizer["PAD_None"], copy_inputs_as_labels=True)
    collator.pad_on_left = True
    collator.eos_token = None
    kwargs_dataset = {
        "max_seq_len": 256, "tokenizer": tokenizer,
        "bos_token_id": tokenizer["BOS_None"],
        "eos_token_id": tokenizer["EOS_None"]
    }
    dataset = DatasetMIDI(midi_paths, **kwargs_dataset)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

def generate_and_save(model, tokenizer, dataloader, generation_config, output_path):
    model.eval()
    for count, batch in enumerate(tqdm(dataloader, desc='Generating results')):
        res = model.generate(
            inputs=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            generation_config=generation_config
        )
        
        for prompt, continuation in zip(batch["input_ids"], res):
            generated = continuation[len(prompt):]
            midi = tokenizer.decode(deepcopy(generated))
            tokens = [generated, prompt, continuation]
            tokens = [seq.tolist() for seq in tokens]
            
            for i, tok_seq in enumerate(tokens[1:], 1):
                _midi = tokenizer.decode(deepcopy(tok_seq))
                midi.tracks.append(_midi.tracks[0])
                midi.tracks[i].name = f'{"Original sample" if i == 1 else "Original sample and continuation"} ({len(tok_seq)} tokens)'
            
            midi.tracks[0].name = f'Continuation of original sample ({len(generated)} tokens)'
            midi.dump_midi(output_path / f'{count}.mid')
            tokenizer.save_tokens(tokens, output_path / f'{count}.json')
            count += 1

def main():
    parser = argparse.ArgumentParser(description="Generate MIDI using a trained model.")
    parser.add_argument("--dataset", type=str, help="Path to directory containing MIDI files")
    args = parser.parse_args()

    model_path = Path("./runs/")
    tokenizer_path = Path("./tokenizer.json")
    output_path = Path('gen_res')
    output_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    generation_config = setup_generation_config(tokenizer)

    if args.dataset:
        midi_paths = list(Path(args.dataset).glob("**/*.mid")) + list(Path(args.dataset).glob("**/*.midi"))
        dataloader = setup_data_loader(midi_paths, tokenizer)
    else:
        dataloader = setup_data_loader([None], tokenizer)

    generate_and_save(model, tokenizer, dataloader, generation_config, output_path)

if __name__ == "__main__":
    main()
