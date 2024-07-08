# 🎹 bach-gpt

bach-gpt is a simple implementation of a Transformer model to learn how to generate MIDI music. 🎵

It uses the MidiTok library to tokenize and prepare the dataset, then trains on the Mistral model with PyTorch and Transformers. 

While it blatantly steals the code examples from the MidiTok GitHub (thanks, MidiTok team! 🙌), here it's implemented and it ~~actually~~ hopefully works! 😄

## 🚀 Features

- MIDI tokenization using MidiTok
- Training on the Mistral Transformer model
- Generation of new MIDI sequences
  
## 🛠 Installation

### Dependencies

Install the required Python packages:

```bash
pip install torch evaluate transformers accelerate tensorboard miditok
```

### 🍏 GPU Support on MacOS

To harness the power of your GPU on MacOS:

1. Install Conda, which comes with a Python version built for Apple Silicon GPUs.
2. Create a specific environment for your project:

```bash
conda create -n myenv
conda activate myenv
```

3. Install the dependencies in this environment.

## 🎵 Usage

### Training the Model

Train the model on your MIDI dataset:

```bash
python src/miditoktok.py --dataset path/to/your/dataset
```

### Generating Music

Generate new MIDI files, with or without prompts:

```bash
python src/generate.py --prompt path/to/your/prompts
```

Leave out the `--prompt` argument to generate from scratch!

## 🤔 How It Works

1. **Tokenization**: MidiTok converts MIDI files into a sequence of tokens that the model can understand.
2. **Training**: The Mistral Transformer model learns patterns from these tokenized sequences.
3. **Generation**: The trained model creates new sequences, which are then converted back into MIDI.

## 🎨 Experimentation Ideas

- Try different model architectures or hyperparameters
- Implement in a VST plugin with UI for DAWs(Digital Audio Workstation).

## 👥 Contributing

Feel free to fork, experiment, and submit pull requests!

## Sources
http://jalammar.github.io/illustrated-transformer/

https://afmck.in/posts/2023-05-22-jax-post/

https://afmck.in/posts/2023-06-04-flax-post/

https://github.com/vvvm23/tchaikovsky

https://github.com/Natooz/MidiTok

