# Generative AI: The Elder Scrolls Character Name Generation

This project is designed as a hands-on learning experience in generative modeling for text data. The application revolves around generating believable NPC (Non-Player Character) names inspired by The Elder Scrolls (TES) game series. This README outlines the project's objectives, structure, and methodologies, as well as gives an outline of how generative AI has progressed over time. This project is in the early stages and is still being being updated.

This project is inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore), which "makes more" human names using autogressive models. This project is different by focusing on TES Lore and character names to "make more lore", and will implement a wider range of techniques. Future versions of this project may also include multimodal data, and generate an image of a character as well.

## Objectives

Gain a deeper understanding of generative modeling and its evolution over time. This project explores the progression of techniques, starting with foundational methods like Markov Models and ending with cutting-edge architectures such as Generative Transformers. Along the way, it will have implementations of notable milestones in generative AI, including RNNs, VAEs, Transformers, and other methods, offering a comprehensive look at how generative models have advanced through the years.

Another objective for this project is to gain experience with webscraping publicly available online information.


## Structure

- `processing/` contains scripts used to obtain and clean any data used in this project
  - `01_scrape_tes_names.R`: uses R's `rvest` package to scrape NPC names from specific game information pages
  - `02_scrape_lore_names.R`: uses R's `rvest` to scrape NPC names from the lore information pages
  - `03_combine_tes_lore.R `: combines the unique information from step `01` and `02`
  - `archive/` stores old code not used anymore
 
- `npc_data.csv`: 43K names from The Elder Scrolls universe. This is web scraped data and is from the output of the code in `processing`
  - `name`: the name of the NPC
  - `sex`: the sex of the NPC
  - `race`: the race of the NPC
  - `url`: a URL to get more information about the NPC
 
- `markov/`  contains a python script to use a Markov Model / N-gram to generate names
- `embedding/` contains python code to generate names using embeddings and a shallow neural network


## Methodologies

[The Unofficial Elder Scrolls Pages](https://en.uesp.net/wiki/Main_Page) has very thorough documentation on all game details including NPC names which were used in this project.

---

## The Evolution of Generative AI

This section gives an overview of how NLP has progressed over the years. It does not go into the details of how each method works, but instead highlights what problems each method addressed and its limitations. Implementations of these methods can be found in code in the related folders.

- [Markov Models / N-grams](#markov-models--n-grams)
- [Embeddings](#embeddings)
- [Recurrent Neural Networks (RNNs)](#recurrent-neural-networks-rnns)
  - [Vanilla RNN](#vanilla-rnn)
  - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
  - [Gated Recurrent Unit (GRU)](#gated-recurrent-unit-gru)
- [Variational Autoencoders (VAEs)](#variational-autoencoders-vaes)
- [Transformers](#transformers)
- [Pretrained Language Models (PLMs)](#pretrained-language-models-plms)
- [Transformer-Based Models](#transformer-based-models)
  - [Large Language Models (LLMs)](#large-language-models-llms)
    - [Zero, One, Few - Shot Learning](#zero-shot-one-shot-few-shot-learning)
    - [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
    - [Fine Tuning](#fine-tuning)
- [State Space Models](#state-space-models)

---

### Markov Models / N-grams

The first language models, known as "n-grams", originated from Claude Shannon's "A Mathematical Theory of Communication" in 1948 [1].

These models took advantage of statistical dependence between nearby words, allowing the model to go beyond simple frequency counts and consider what came before.

Limitations:
1. The number of possible n-grams grows exponentially as $n$ increases, leading to data sparsity ("curse of dimensionality").
2. N-grams only consider the preceding $n−1$ words, making them blind to long-range dependencies.

**References**
- [[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6773024) *A Mathematical Theory of Communication* – Shannon, C. (1948)

---

### Embeddings

- [[1]](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) *A Neural Probabilistic Language Model* – Bengio, Y. et al. (2003)
- [[2]](https://arxiv.org/abs/1301.3781) *Efficient Estimation of Word Representations in Vector Space (Word2Vec)* – Mikolov, T. et al. (2013)
- [[3]](https://aclanthology.org/D14-1162.pdf) *GloVe: Global Vectors for Word Representation* – Pennington, J. et al. (2014)

---

### Recurrent Neural Networks (RNNs)

#### Vanilla RNN
- [[1]](https://www.fit.vut.cz/research/group/speech/public/publi/2010/mikolov_interspeech2010_IS100722.pdf) *Recurrent Neural Network Based Language Model* – Mikolov, T. et al. (2010)

#### Long Short-Term Memory (LSTM)
- [[1]](https://www.bioinf.jku.at/publications/older/2604.pdf) *Long Short-Term Memory* – Hochreiter, S. and Schmidhuber, J. (1997)
- [[2]](https://arxiv.org/abs/1409.3215) *Sequence to Sequence Learning with Neural Networks* – Sutskever, I. et al. (2014)
- [[3]](https://arxiv.org/abs/1308.0850) *Generating Sequences With Recurrent Neural Networks* – Graves, A. (2013)

---

### Variational Autoencoders (VAEs)
- [[1]](https://arxiv.org/abs/1511.06349) *Generating Sentences from a Continuous Space* – Bowman, S. et al. (2016)

---

### Transformers

- [[1]](https://arxiv.org/abs/1409.0473) *Neural Machine Translation by Jointly Learning to Align and Translate* – Bahdanau, D. et al. (2014)
- [[2]](https://arxiv.org/abs/1706.03762) *Attention Is All You Need* – Vaswani, A. et al. (2017)

---

### Pretrained Language Models (PLMs)
- [[1]](https://arxiv.org/abs/1802.05365) *Deep Contextualized Word Representations (ELMo)* – Peters, M. et al. (2018)
- [[2]](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) *Improving Language Understanding by Generative Pre-Training (GPT-1)* – Radford, A. et al. (2018)
- [[3]](https://arxiv.org/abs/1810.04805) *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* – Devlin, J. et al. (2018)
- [[5]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) *Language Models are Unsupervised Multitask Learners (GPT-2)* – Radford, A. et al. (2019)

### Transformer-Based Models

#### Large Language Models (LLMs)
- [[2]](https://arxiv.org/abs/2203.02155) *Training Language Models to Follow Instructions with Human Feedback (InstructGPT)* – Ouyang, L. et al. (2022)
- [[3]](https://arxiv.org/abs/2302.13971) *LLaMA: Open and Efficient Foundation Language Models* – Touvron, H. et al. (2023)

##### Zero-shot, One-shot, Few-shot Learning
- [[1]](https://arxiv.org/abs/2005.14165) *Language Models are Few-Shot Learners (GPT-3)* – Brown, T. et al. (2020)

##### Retrieval Augmented Generation (RAG)
- [[1]](https://arxiv.org/abs/2005.11401) *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* – Lewis, P. et al. (2020)

##### Fine Tuning
- [[1]](https://arxiv.org/abs/2106.09685) *LoRA: Low-Rank Adaptation of Large Language Models* – Hu, E. et al. (2021)
- [[2]](https://arxiv.org/abs/2305.14314) *QLoRA: Efficient Finetuning of Quantized LLMs* – Dettmers, T. et al. (2023)

---

### State Space Models
- [[1]](https://arxiv.org/abs/2312.00752) *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* – Gu, A. and Dao, T. (2023)
- [[2]](https://arxiv.org/abs/2111.00396) *Combining Recurrent, Convolutional, and Attention Mechanisms Through State Space Models (S4)* – Gu, A. et al. (2021)

---
