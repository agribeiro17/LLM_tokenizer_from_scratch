# LLM Tokenizer from Scratch

This project walks through the process of building a simple text tokenizer from scratch in Python. The implementation is detailed in the `LLMTokenizer.ipynb` notebook and uses a short story, "The Verdict" by Edith Wharton, as the training corpus.

## Project Overview

The goal of this project is to understand the fundamental principles of tokenization, a core component of Large Language Models (LLMs). The process involves converting raw text into a sequence of numerical token IDs that can be fed into a model, and then decoding those IDs back into human-readable text.

## Key Concepts and Implementation Steps

The notebook is structured in the following steps:

### 1. Text Preprocessing and Initial Tokenization

- The `the-verdict.txt` file is read as the raw text corpus.
- A simple tokenization scheme is implemented using regular expressions (`re.split`) to split the text into a list of tokens. The process starts by splitting on whitespace and is progressively refined to also treat punctuation as separate tokens.

### 2. Vocabulary Creation

- A vocabulary is built by collecting all unique tokens from the preprocessed text.
- The vocabulary is sorted alphabetically.

### 3. Token ID Assignment

- Each unique token in the vocabulary is assigned a unique integer ID. This is done by creating a dictionary that maps each string token to an integer.

### 4. `SimpleTokenizerV1`

- A `SimpleTokenizerV1` class is implemented to encapsulate the tokenization logic.
- **`encode(text)`**: This method takes a string, preprocesses it according to the defined tokenization rules, and converts it into a list of token IDs using the vocabulary.
- **`decode(ids)`**: This method takes a list of token IDs and converts them back into a string. It also includes logic to clean up extra spaces around punctuation.

### 5. Handling Out-of-Vocabulary Words

- The limitation of `SimpleTokenizerV1` is demonstrated when it encounters a word not present in its vocabulary, resulting in a `KeyError`.

### 6. `SimpleTokenizerV2` with Special Tokens

- To address the out-of-vocabulary issue, a `SimpleTokenizerV2` is introduced.
- This version enhances the vocabulary with two special tokens:
    - `<|unk|>`: Used to represent any token that is not found in the vocabulary.
    - `<|endoftext|>`: A special token used to signal the end of a text segment, which is useful when concatenating multiple documents.
- The `encode` method is updated to map any unknown word to the `<|unk|>` token's ID instead of raising an error.

## How to Use

1.  Open the `LLMTokenizer.ipynb` notebook in a Jupyter environment.
2.  Ensure you have the `the-verdict.txt` file in the same directory.
3.  Run the cells sequentially to follow the step-by-step process of building and testing the tokenizers.

## Future Directions

The notebook concludes with a discussion of other special tokens used in LLMs, such as `[BOS]`, `[EOS]`, and `[PAD]`, and mentions that more advanced tokenizers like Byte Pair Encoding (BPE) are used in models like GPT to handle the out-of-vocabulary problem more effectively by breaking words into subword units.

## Byte Pair Encoding (BPE)

The notebook also introduces Byte Pair Encoding (BPE), a more advanced tokenization technique used in models like GPT. BPE is a subword-based tokenization algorithm that iteratively merges the most frequent pair of bytes in the vocabulary. This allows the tokenizer to handle any word, including unknown words, by breaking them down into smaller, known subword units.

### Using `tiktoken` for BPE

The `tiktoken` library from OpenAI is used to demonstrate BPE in practice. It provides a simple way to use the same tokenizer as GPT models.

**Installation:**
```bash
pip3 install tiktoken
```

**Encoding and Decoding with `tiktoken`:**
```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunkownPlace.")

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)
```

This demonstrates how BPE can encode and decode text, including out-of-vocabulary words like "someunkownPlace", without needing an `<|unk|>` token. The word is simply broken down into subword units that are in the vocabulary.

## Creating Input-Target Pairs for Language Modeling

After tokenizing the text, the next step is to prepare the data for training a language model. This involves creating input-target pairs. For a next-word prediction task, the input is a sequence of tokens, and the target is the next token in the sequence.

The notebook demonstrates a sliding window approach to generate these pairs from the tokenized text. For a given `context_size`, the input `x` is a sequence of tokens of that length, and the target `y` is the sequence shifted by one position.

```python
context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]

print(f"x: {x}")
print(f"y: {y}")
```

## Data Loading with PyTorch

To efficiently load the data in batches for training, the notebook uses PyTorch's `Dataset` and `DataLoader` classes.

### `GPTDatasetV1`

A custom `GPTDatasetV1` class is implemented, which inherits from `torch.utils.data.Dataset`. This class:
- Takes the tokenized text, a tokenizer, a `max_length` (context size), and a `stride` as input.
- Uses a sliding window to create chunks of `max_length` from the text.
- For each chunk, it creates an `input_chunk` and a `target_chunk` (shifted by one).
- These chunks are stored as PyTorch tensors.

```python
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

### `create_dataloader_v1`

A helper function `create_dataloader_v1` is created to instantiate the `GPTDatasetV1` and wrap it in a `DataLoader`. The `DataLoader` provides an iterator for easy batching, shuffling, and parallel data loading.

```python
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
```

This setup allows for efficient training of a language model by providing a stream of input-target batches.