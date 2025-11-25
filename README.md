# LLM Tokenizer from Scratch

This project walks through the process of building a simple text tokenizer from scratch in Python. The implementation is detailed in the `LLMTokenizer.ipynb` notebook and uses a short story, "The Verdict" by Edith Wharton, as the training corpus.

## Project Overview

The goal of this project is to understand the fundamental principles of tokenization, a core component of Large Language Models (LLMs). The process involves converting raw text into a sequence of numerical token IDs that can be fed into a model, and then decoding those IDs back into human-readable text.

## How to Use

1.  Open the `LLMTokenizer.ipynb` notebook in a Jupyter environment.
2.  Ensure you have the `the-verdict.txt` file in the same directory.
3.  Run the cells sequentially to follow the step-by-step process of building and testing the tokenizers.

## Key Concepts and Implementation Steps

The `LLMTokenizer.ipynb` notebook is structured in the following steps:

### Text Preprocessing and Initial Tokenization
- The `the-verdict.txt` file is read as the raw text corpus.
- A simple tokenization scheme is implemented using regular expressions (`re.split`) to split the text into a list of tokens. The process starts by splitting on whitespace and is progressively refined to also treat punctuation as separate tokens.

### Vocabulary Creation
- A vocabulary is built by collecting all unique tokens from the preprocessed text.
- The vocabulary is sorted alphabetically.

### Token ID Assignment
- Each unique token in the vocabulary is assigned a unique integer ID. This is done by creating a dictionary that maps each string token to an integer.

### SimpleTokenizerV1
- A `SimpleTokenizerV1` class is implemented to encapsulate the tokenization logic.
- Its `encode(text)` method takes a string, preprocesses it according to the defined tokenization rules, and converts it into a list of token IDs using the vocabulary.
- Its `decode(ids)` method takes a list of token IDs and converts them back into a string, including logic to clean up extra spaces around punctuation.

### Handling Out-of-Vocabulary Words
- The limitation of `SimpleTokenizerV1` is demonstrated when it encounters a word not present in its vocabulary, resulting in a `KeyError`.

### SimpleTokenizerV2 with Special Tokens
- To address the out-of-vocabulary issue, a `SimpleTokenizerV2` is introduced.
- This version enhances the vocabulary with two special tokens: `<|unk|>` (for unknown tokens) and `<|endoftext|>` (to signal the end of a text segment).
- The `encode` method is updated to map any unknown word to the `<|unk|>` token's ID instead of raising an error.

## Byte Pair Encoding (BPE)

This section introduces Byte Pair Encoding (BPE), a more advanced tokenization technique used in models like GPT. BPE is a subword-based tokenization algorithm that iteratively merges the most frequent pair of bytes in the vocabulary. This allows the tokenizer to handle any word, including unknown words, by breaking them down into smaller, known subword units.

### Using `tiktoken` for BPE

The `tiktoken` library from OpenAI is used to demonstrate BPE in practice. The implementation involves using the `get_encoding` function to obtain a tokenizer, and then utilizing its `encode` and `decode` methods to convert text to tokens and back. This effectively showcases how BPE handles out-of-vocabulary words by segmenting them into subword units.

## Creating Input-Target Pairs for Language Modeling

After tokenizing the text, the next step is to prepare the data for training a language model. This involves creating input-target pairs. For a next-word prediction task, the input is a sequence of tokens, and the target is the next token in the sequence. The notebook demonstrates a sliding window approach to generate these pairs from the tokenized text.

## Data Loading with PyTorch

To efficiently load the data in batches for training, the notebook uses PyTorch's `Dataset` and `DataLoader` classes.

### `GPTDatasetV1`

A custom `GPTDatasetV1` class is implemented, which inherits from `torch.utils.data.Dataset`. This class takes the tokenized text, a tokenizer, a `max_length` (context size), and a `stride` as input. It uses a sliding window to create input-target chunks and stores them as PyTorch tensors.

### `create_dataloader_v1`

A helper function `create_dataloader_v1` is created to instantiate the `GPTDatasetV1` and wrap it in a `DataLoader`. The `DataLoader` provides an iterator for easy batching, shuffling, and parallel data loading.

## Token Embeddings

Token embeddings are a fundamental concept in modern Natural Language Processing (NLP). After tokenizing text into numerical IDs, these IDs are converted into dense vector representations called embeddings. Each token ID is mapped to a vector that captures semantic and syntactic information.

This concept was explored in the `LLMTokenizer.ipynb` notebook. To provide a more hands-on demonstration, a new notebook, `Vector_Embedding.ipynb`, was created.

### Vector Embedding Demo (`Vector_Embedding.ipynb`)

The `Vector_Embedding.ipynb` notebook demonstrates the power of word embeddings using the `gensim` library and a pre-trained `word2vec` model from Google. The key steps in the notebook are:

*   **Loading a Pre-trained Model:** It loads the `word2vec-google-news-300` model, which contains vector representations for a vast number of words.
*   **Exploring Word Vectors:** It shows how to access the 300-dimensional vector for any word in the model's vocabulary.
*   **Measuring Semantic Similarity:** The notebook demonstrates how to calculate the similarity between pairs of words (e.g., "king" and "queen") and find the most similar words to a given word.
*   **Vector Arithmetic:** It showcases the famous "king" + "woman" - "man" = "queen" example to illustrate how the vector space captures semantic relationships.
*   **Vector Difference:** The script also calculates the magnitude of the difference between word vectors to quantify their semantic distance.

This notebook provides a clear, practical look at how word embeddings work and how they can be used to understand the relationships between words.

## Positional Embeddings

In addition to token embeddings, which capture the semantic meaning of words, transformer-based models like GPT require a way to understand the order of words in a sequence. This is achieved through **positional embeddings**. Since the self-attention mechanism in transformers processes all tokens in parallel, it has no inherent sense of sequence order. Positional embeddings provide this crucial context by adding a vector to each token embedding that represents its position in the input sequence.

The `LLMTokenizer.ipynb` notebook demonstrates this concept by:

*   **Creating a Positional Embedding Layer:** A new `torch.nn.Embedding` layer is created specifically for positional embeddings. Its size is determined by the `context_length` (the maximum length of an input sequence) and the `output_dim` (the embedding dimension, which matches the token embedding dimension).
*   **Generating Positional Embeddings:** A sequence of numbers from `0` to `context_length - 1` is created using `torch.arange(context_length)`. This sequence is then passed to the positional embedding layer to get the corresponding positional embedding vectors.
*   **Combining with Token Embeddings:** The resulting positional embeddings are added directly to the token embeddings. PyTorch's broadcasting capabilities ensure that the positional embedding tensor is added to each sequence in the batch, resulting in a final input embedding tensor that contains both semantic and positional information. This combined embedding is then ready to be processed by the main LLM modules.

## Future Directions

The notebook concludes with a discussion of other special tokens used in LLMs, suchs as `[BOS]`, `[EOS]`, and `[PAD]`, and mentions that more advanced tokenizers like Byte Pair Encoding (BPE) are used in models like GPT to handle the out-of-vocabulary problem more effectively by breaking words into subword units.