library(rvest)
library(tidyverse)

source('Data_Processing/utils.R')

# Web scrape character names from the elder scrolls (TES) games
# https://en.uesp.net/wiki/Main_Page

# They seem to have most of the names and links to profiles here: https://en.uesp.net/wiki/Lore:Names
# The breakdown is by race and sex


#### Argonian ####
url <- 'https://en.uesp.net/wiki/Lore:Argonian_Names'

# Scrape names with a link to their profile
argonian_all_linked_names <- scrape_all(url, 'p a')
last_male_name_link <- 'Scale-Song'

# Scrape names without a profile
argonian_all_nonlinked_names <- scrape_data(url, 'p:nth-child(68) , p:nth-child(39) , p+ table td')
last_male_name_nonlink <- 'Water-Chaser'
argonian_all_nonlinked_names <- clean_strings(argonian_all_nonlinked_names)

# Add sex
# website has names ordered by sex, everyone before a certain name is male
unlinked <- tibble(name = argonian_all_nonlinked_names) %>% 
  mutate(id = row_number()) %>% 
  mutate(sex = ifelse(id <= .[.$name == last_male_name_nonlink,]$id, 'Male', 'Female')) %>% 
  select(-id)

linked <- as_tibble(argonian_all_linked_names) %>% 
  filter(!(name %in% c('[1]', '[2]', '[3]', '[4]', '[5]', '†', 'Argonian', 'Jel', 'Hist Sap')))

# Bring over full name
# Some names were linked to stories about them, not their actual profile
# For these people we do not want to use their full name
# Some times two people had the same name but were different - these were pulled as a number (i.e., 1 and 2)
# This code will duplicate people -> give us their first name, then make another row with first name last name
linked <- linked %>% 
  mutate(full_name = str_remove(full_name, '( \\(.*\\))$')) %>% 
  filter(str_detect(full_name, name) | name %in% c('1', '2')) %>% 
  filter(name != full_name) %>% 
  mutate(name = full_name) %>% 
  bind_rows(linked) %>% 
  select(-full_name) %>%
  distinct %>% 
  filter(name != '1')

# Add sex
# website has names ordered by sex, everyone before a certain name is male
linked <- linked %>% 
  mutate(id = row_number()) %>% 
  mutate(sex = ifelse(id <= .[.$name == last_male_name_link,]$id, 'Male', 'Female')) %>% 
  select(-id)

argonians <- linked %>% bind_rows(unlinked) %>% mutate(race = 'Argonian') %>% distinct %>% 
  select(name, sex, race, url)
