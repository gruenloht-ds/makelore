# Generative AI: The Elder Scrolls Character Name Generation

This project is designed as a hands-on learning experience in generative modeling for text data. The application revolves around generating believable NPC (Non-Player Character) names inspired by The Elder Scrolls (TES) game series. This README outlines the project's objectives, structure, and methodologies, as well as gives an outline of how generative AI has progressed over time. This project is in the early stages and is still being being updated.

## Objectives

Gain a deeper understanding of generative modeling and its evolution over time. This project explores the progression of techniques, starting with foundational methods like Hidden Markov Models and ending with cutting-edge architectures such as Generative Transformers. Along the way, it will have implementations of notable milestones in generative AI, including RNNs, VAEs, RBMs, and others, offering a comprehensive look at how generative models have advanced through the years.

A secondary objective for this project is to gain experience with webscraping publicly available online information.


## Structure

- `Data_Processing` contains scripts used to obtain, clean, and augment any data used in this project
  - `01_scrape_tes_names.R` uses R's `rvest` package to scrape NPC names
  - `02_clean_npc_names.R` cleans, removes, and formats npc names to reflect webpage URLs better


## Methodologies

[The Unofficial Elder Scrolls Pages](https://en.uesp.net/wiki/Main_Page) has very thorough documentation on all game details including NPC names which were used in this project.

---

## The Evlolution of Generative AI

### Markov Models

### Embeddings

### Recurrent Neural Networks (RNNs)

##### Vanilla RNN

##### Long Short-Term Memory (LSTM)

##### Gated Recurrent Unit (GRU)

### Variational Autoencoders (VAEs)

### Restricted Boltzmann Machines (RBMs)

### Generative Adversarial Networks (GANs)

### Transformers

### Pretrained LLMs

##### Zero-shot

##### Few-shot

##### Retreival Augmented Generation (RAG)

##### Fine Tunning

### State Space Models
