# Generative AI: The Elder Scrolls Character Name Generation

This project is designed as a hands-on learning experience in generative modeling for text data. The application revolves around generating believable NPC (Non-Player Character) names inspired by The Elder Scrolls (TES) game series. This README outlines the project's objectives, structure, and methodologies, as well as gives an outline of how generative AI has progressed over time. This project is in the early stages and is still being being updated.

This project is inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore), which "makes more" human names using autogressive models. This project is different by focusing on TES Lore and character names to "make more lore", and will implement a wider range of techniques.

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

### Markov Models
- [[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6773024)
### Embeddings
- [[1]](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
### Recurrent Neural Networks (RNNs)

##### Vanilla RNN
-[[1]](https://icml.cc/2011/papers/524_icmlpaper.pdf)
##### Long Short-Term Memory (LSTM)
- [[1]](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [[2]](https://arxiv.org/abs/1409.3215)
##### Gated Recurrent Unit (GRU)
-[[1]](https://arxiv.org/abs/1406.1078)
### Variational Autoencoders (VAEs)
 - [[1]](/https://arxiv.org/pdf/1511.06349)

### Restricted Boltzmann Machines (RBMs)

### Generative Adversarial Networks (GANs)

### Transformers
- [attention](https://arxiv.org/pdf/1409.0473)
- [[2]](https://arxiv.org/pdf/1706.03762)

### Pretrained LLMs
- [[1]](https://arxiv.org/pdf/1810.04805)
##### Zero-shot

##### Few-shot
- [[1]](https://arxiv.org/pdf/2005.14165)
##### Retreival Augmented Generation (RAG)

##### Fine Tunning

### State Space Models
- [[1]](https://arxiv.org/abs/2312.00752)
