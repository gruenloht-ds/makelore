# Generative AI: The Elder Scrolls Character Name Generation

This project is designed as a hands-on learning experience in generative modeling for text data. The application revolves around generating believable NPC (Non-Player Character) names inspired by The Elder Scrolls (TES) game series. This README outlines the project's objectives, structure, and methodologies, as well as gives an outline of how generative AI has progressed over time. This project is in the early stages and is still being being updated.

This project is inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore), which "makes more" human names using autogressive models. This project is different by focusing on TES Lore and character names to "make more lore", and will implement a wider range of techniques. Future versions of this project may also include multimodal data, and generate an image of a character as well.

## Objectives

Gain a deeper understanding of generative modeling and its evolution over time. This project explores the progression of techniques, starting with foundational methods like Markov Models and ending with cutting-edge architectures such as Generative Transformers. Along the way, it will have implementations of notable milestones in generative AI, including RNNs, VAEs, RBMs, and others, offering a comprehensive look at how generative models have advanced through the years.

A secondary objective for this project is to gain experience with webscraping publicly available online information.


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
 
- `markov/`  contains python script to use a Markov Model / N-gram to generate names


## Methodologies

[The Unofficial Elder Scrolls Pages](https://en.uesp.net/wiki/Main_Page) has very thorough documentation on all game details including NPC names which were used in this project.

---

## The Evlolution of Generative AI
This section is ment to give an overview of how NLP has progressed over the years. It is not meant to give descriptions of how these methods work. Rather, state what problems the method solved and what were some of the limitations. Implementations fo these methods can be found in code in the related folders.

### Markov Models / N-grams

The first language models, known as "n-grams", originated from Claude Shannon's "A Mathematical Theory of Communication" in 1948 [1].

These models took advantage of higher statistical dependence between words (or characters) that are closer in a sentence, allowing the model to go beyond simple frequency counts and consider what was already said. 

The limitations of n-grams include two main drawbacks:

1. As $n$ increases, the number of possible n-grams grows exponentially. This is often referred to as the "curse of dimensionality", and requires vast amounts of data to accurately estimate the probabilities of all n-grams.
2. N-grams only consider the preceding $n−1$ words when making predictions. However, natural language often contains long-range dependencies/connections between words or phrases in a sentence or paragraph.


References
- [[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6773024)

### Embeddings
- [[1]](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
### Recurrent Neural Networks (RNNs)

##### Vanilla RNN
- [[1]](https://icml.cc/2011/papers/524_icmlpaper.pdf)
##### Long Short-Term Memory (LSTM)
- [[1]](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [[2]](https://arxiv.org/abs/1409.3215)
##### Gated Recurrent Unit (GRU)
- [[1]](https://arxiv.org/abs/1406.1078)
### Variational Autoencoders (VAEs)
- [[1]](https://arxiv.org/abs/1511.06349)

### Restricted Boltzmann Machines (RBMs)

### Generative Adversarial Networks (GANs)

### Transformers
- [[1]](https://arxiv.org/abs/1409.0473): attention calculated with MLP
- [[2]](https://arxiv.org/abs/1508.04025): dot product attention (unscaled)
- [[3]](https://arxiv.org/abs/1706.03762): transformer architecture 
### Pretrained LLMs
- [[1]](https://arxiv.org/abs/1810.04805)
##### Zero-shot

##### Few-shot
- [[1]](https://arxiv.org/abs/2005.14165)
##### Retreival Augmented Generation (RAG)
- [[1]](https://arxiv.org/abs/2005.11401)
##### Fine Tunning

### State Space Models
- [[1]](https://arxiv.org/abs/2312.00752)
