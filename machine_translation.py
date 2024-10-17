

!pip install torchtext==0.6

# Install the German language model for spaCy
!python -m spacy download de_core_news_sm

# Install the English language model for spaCy
!python -m spacy download en_core_web_sm

import spacy
import torch
import torchtext
from torchtext.data import Field, Dataset, Example
from torchtext.data import BucketIterator

# Load the spacy models for German and English
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

# Tokenization functions
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# Fields for German and English
german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

import os

# Function to load data from files
def load_data(german_file, english_file, german_field, english_field):
    # Open the files
    with open(german_file, 'r', encoding='utf-8') as f:
        german_sentences = f.readlines()

    with open(english_file, 'r', encoding='utf-8') as f:
        english_sentences = f.readlines()

    # Ensure they are the same length
    assert len(german_sentences) == len(english_sentences), "Mismatch in sentence count!"

    # Create examples
    examples = []
    for german_sentence, english_sentence in zip(german_sentences, english_sentences):
        # Create an example from the German and English sentences
        example = Example.fromlist([german_sentence.strip(), english_sentence.strip()], fields=[('src', german_field), ('trg', english_field)])
        examples.append(example)

    # Create a dataset from examples
    dataset = Dataset(examples, fields=[('src', german_field), ('trg', english_field)])
    return dataset

# File paths (adjust to your file locations)
train_ger_path = "/content/drive/MyDrive/dataset/train.de"
train_eng_path = "/content/drive/MyDrive/dataset/train.en"
val_ger_path = "/content/drive/MyDrive/dataset/val.de"
val_eng_path = "/content/drive/MyDrive/dataset/val.en"
test_ger_path = "/content/drive/MyDrive/dataset/test_2016_flickr.de"
test_eng_path = "/content/drive/MyDrive/dataset/test_2016_flickr.en"

# Load the datasets
train_data = load_data(train_ger_path, train_eng_path, german, english)
valid_data = load_data(val_ger_path, val_eng_path, german, english)
test_data = load_data(test_ger_path, test_eng_path, german, english)

# Build the vocabulary for German and English
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

# Create BucketIterators for batching
BATCH_SIZE = 32  # Adjust as needed
train_iterator = BucketIterator(train_data, batch_size=BATCH_SIZE, sort_within_batch=True, sort_key=lambda x: len(x.src), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
valid_iterator = BucketIterator(valid_data, batch_size=BATCH_SIZE, sort_within_batch=True, sort_key=lambda x: len(x.src), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
test_iterator = BucketIterator(test_data, batch_size=BATCH_SIZE, sort_within_batch=True, sort_key=lambda x: len(x.src), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

import torch.nn as nn

class Transformer_NMT(nn.Module):
    def __init__(self, embedding_dim, src_vocab_size, trg_vocab_size, n_heads, n_layers, src_pad_idx, ff_dim, max_len, dropout, device):
        super(Transformer_NMT, self).__init__()
        self.src_tok_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.src_pos_embedding = nn.Embedding(max_len, embedding_dim)
        self.trg_tok_embedding = nn.Embedding(trg_vocab_size, embedding_dim)
        self.trg_pos_embedding = nn.Embedding(max_len, embedding_dim)
        self.device = device

        self.transformer = nn.Transformer(
            d_model = embedding_dim,
            nhead = n_heads,
            num_encoder_layers = n_layers,
            num_decoder_layers = n_layers,
            dim_feedforward = ff_dim,
            dropout = dropout,
            )

        # output of transformer model is: [target_seq_length, batch_size, hid_dim=embedding_dim]
        self.fc_out = nn.Linear(embedding_dim, trg_vocab_size)
        # we are transformering it to get: [target_seq_length, batch_size, output_dim=trg_vocb_size]

        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx # this is to tell the model which tokens in src should be ignored (as it is a pad token)

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx # creating a BoolTensor
        return src_mask.to(self.device)
        # so essentially we are telling model to ignore the src positions which have pad token

    def forward(self, src, trg):
        src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N).to(self.device)
        ) # here expand will be expanded to a larger size
        trg_positions = (
            torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N).to(self.device)
        )

        src_embedding = self.dropout(self.src_tok_embedding(src) + self.src_pos_embedding(src_positions))
        trg_embedding = self.dropout(self.trg_tok_embedding(trg) + self.trg_pos_embedding(trg_positions))

        src_pad_mask = self.make_src_mask(src)
        # print(trg_seq_len)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)
        # print(trg_mask.shape)

        output = self.transformer(
            src = src_embedding,
            tgt = trg_embedding,
            src_key_padding_mask = src_pad_mask,
            tgt_mask = trg_mask,
        )
        output = self.fc_out(output)

        return output

import torch
import torch.optim as optim
import torch.nn as nn

# Parameters
EMBEDDING_DIM = 256
SRC_VOCAB_SIZE = len(german.vocab)  # Vocabulary size for German
TRG_VOCAB_SIZE = len(english.vocab)  # Vocabulary size for English
N_HEADS = 8  # Number of attention heads
N_LAYERS = 3  # Number of transformer layers
FF_DIM = 512  # Dimension of feedforward layers
MAX_LEN = 100  # Maximum length of sequences
DROPOUT = 0.1
SRC_PAD_IDX = german.vocab.stoi[german.pad_token]  # Pad token index
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = Transformer_NMT(
    embedding_dim=EMBEDDING_DIM,
    src_vocab_size=SRC_VOCAB_SIZE,
    trg_vocab_size=TRG_VOCAB_SIZE,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    src_pad_idx=SRC_PAD_IDX,
    ff_dim=FF_DIM,
    max_len=MAX_LEN,
    dropout=DROPOUT,
    device=DEVICE,
)

# Move the model to the correct device
model.to(DEVICE)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=SRC_PAD_IDX)  # Ignore pad tokens in loss calculation
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adjust learning rate as needed

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Function to calculate perplexity
def calculate_perplexity(loss):
    return np.exp(loss)

def train_with_early_stopping(model, train_iterator, valid_iterator, optimizer, criterion, patience, max_epochs, device):
    best_valid_perplexity = float('inf')  # Keep track of the best perplexity
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # Train
        train_loss = train(model, train_iterator, optimizer, criterion, device)

        # Validate
        valid_loss = evaluate(model, valid_iterator, criterion, device)
        valid_perplexity = calculate_perplexity(valid_loss)

        # Check if the perplexity improves
        if valid_perplexity < best_valid_perplexity:
            best_valid_perplexity = valid_perplexity
            torch.save(model.state_dict(), 'best-model.pt')  # Save the best model
            epochs_without_improvement = 0  # Reset the patience counter
        else:
            epochs_without_improvement += 1

        # Print progress
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.3f}, Validation Loss = {valid_loss:.3f}, Validation Perplexity = {valid_perplexity:.3f}")

        # Early stopping condition
        if epochs_without_improvement >= patience:
            print("Early stopping due to no improvement in validation perplexity.")
            break

    # Return the best model
    return best_valid_perplexity

# Hyperparameters
MAX_EPOCHS = 5 # Maximum number of epochs to train
PATIENCE = 5  # Number of epochs to wait before early stopping

# Train the model with early stopping
best_perplexity = train_with_early_stopping(
    model,
    train_iterator,
    valid_iterator,
    optimizer,
    criterion,
    PATIENCE,
    MAX_EPOCHS,
    DEVICE
)

print(f"Best Validation Perplexity: {best_perplexity:.3f}")

# Load the best model
model.load_state_dict(torch.load('best-model.pt'))

# Evaluate on the test set
test_loss = evaluate(model, test_iterator, criterion, DEVICE)
test_perplexity = calculate_perplexity(test_loss)

print(f"Test Perplexity: {test_perplexity:.3f}")

# Install sacrebleu
!pip install sacrebleu

import torch
import torch.nn.functional as F

def translate_sentence(model, src, german_field, english_field, device, max_length=50):
    # Define start and end tokens
    sos_idx = english_field.vocab.stoi["<sos>"]
    eos_idx = english_field.vocab.stoi["<eos>"]

    # If `src` is a tensor, convert it to text tokens
    if isinstance(src, torch.Tensor):
        tokens = [german_field.vocab.itos[idx] for idx in src]
    else:
        tokens = german_field.tokenize(src)  # Tokenize input

    # Add start and end tokens
    tokens = ["<sos>"] + tokens + ["<eos>"]

    # Convert tokens to tensor indices
    src_indexes = [german_field.vocab.stoi[token] for token in tokens]

    # Convert to tensor and add a batch dimension
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    # Initialize the target with the start token
    trg_indexes = [sos_idx]

    for i in range(max_length):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(device)

        # Get model output
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)

        # Get the logits for the last token
        next_token_logits = output[-1, :]

        # Choose the most probable token
        next_token = torch.argmax(F.softmax(next_token_logits, dim=1)).item()

        trg_indexes.append(next_token)

        # Stop if we reach the end token
        if next_token == eos_idx:
            break

    # Convert token indices back to text
    trg_tokens = [english_field.vocab.itos[i] for i in trg_indexes]

    # Exclude start and end tokens
    return trg_tokens[1:-1]  # Return the generated translation

import sacrebleu
import torch

def compute_bleu_score(model, test_iterator, german, english, device):
    model.eval()  # Set model to evaluation mode
    references = []  # Initialize as a list of list of strings
    hypotheses = []  # Initialize as a list of strings

    with torch.no_grad():  # Ensure no gradients are computed during inference
        for batch in test_iterator:
            src = batch.src.to(device)  # German source sentences
            trg = batch.trg.to(device)  # English target sentences (references)

            # Iterate over each sentence in the batch
            for i in range(src.shape[1]):
                # Translate the German sentence
                translation = translate_sentence(
                    model, src[:, i], german, english, device, max_length=50
                )

                # Get the reference translation as a list of strings
                trg_sentence = [english.vocab.itos[word] for word in trg[1:, i]]

                # Ensure the structure is a list of list of strings
                references.append([trg_sentence])

                # Hypotheses should be a single string (joined tokens)
                hypotheses.append(" ".join(translation))

    # Check that references and hypotheses have correct structure
    assert isinstance(references, list), "References should be a list"
    assert all(isinstance(ref, list) for ref in references), "Each reference should be a list of strings"
    assert all(isinstance(hyp, str) for hyp in hypotheses), "Hypotheses should be a list of strings"

    # Compute BLEU score
    bleu_score = sacrebleu.corpus_bleu(hypotheses, references)

    return bleu_score.score

