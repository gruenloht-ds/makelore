library(rvest)
library(tidyverse)

source('Data_Processing/utils.R')

# Web scrape character names from the elder scrolls (TES) games
# https://en.uesp.net/wiki/Main_Page

# They seem to have most of the names and links to profiles here: https://en.uesp.net/wiki/Lore:Names
# The breakdown is by race and sex

# There are two types of names on this page, 
#   1. Names with links to profiles/character information
#   2. Names without profiles/character information or links to such info

# As the structure of the names on each page is roughly the same, the code in each section will be roughly the same aswell

#### Altmer Names ####
url <- 'https://en.uesp.net/wiki/Lore:Altmer_Names'

# Scrape names with a link to their profile
linked_names <- scrape_all(url, 'p a')
last_male_name_link <- 'Vorian'
last_female_name_link <- 'Tuinden'

# Scrape names without a profile
nonlinked_names <- scrape_data(url, 'p:nth-child(24) , p:nth-child(49) , p+ table td')
last_male_name_nonlink <- 'Yarelion'
last_female_name_nonlink <- 'Yamanwe'

nonlinked_names <- clean_strings(nonlinked_names)

# Add sex
# website has names ordered by sex, everyone before a certain name is male
unlinked <- tibble(name = nonlinked_names) %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_nonlink) ~ 'Male',
      id <= which(name == last_female_name_nonlink) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

linked <- as_tibble(linked_names) %>% 
  filter(!(name %in% c('[1]', '[2]', 'Dunmer name', 'Altmer')))

# Add sex
# website has names ordered by sex, everyone before a certain name is male
linked <- linked %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_link) ~ 'Male',
      id <= which(name == last_female_name_link) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

# Bring over full name
# Some names were linked to stories about them, not their actual profile
# For these people we do not want to use their full name
# Some times two people had the same name but were different - these were pulled as a number (i.e., 1 and 2)
# This code will duplicate people -> give us their first name, then make another row with first name last name
linked <- linked %>% 
  mutate(full_name = str_remove(full_name, '( \\(.*\\))$')) %>% 
  filter(str_detect(full_name, name) | !is.na(as.numeric(name))) %>% 
  filter(name != full_name) %>% 
  mutate(name = full_name) %>% 
  bind_rows(linked) %>% 
  select(-full_name) %>%
  distinct %>% 
  filter(is.na(as.numeric(name))) # Remove any numeric names (only referenced when there were multiple same names)

# Create final list of names
altmer <- linked %>% 
  bind_rows(unlinked) %>% 
  mutate(race = 'Altmer') %>% 
  distinct(name, sex, race, url)


#### Argonian Names ####
url <- 'https://en.uesp.net/wiki/Lore:Argonian_Names'

# Scrape names with a link to their profile
linked_names <- scrape_all(url, 'p a')
last_male_name_link <- 'Scale-Song'
last_female_name_link <- 'Sheen-in-Glade'

# Scrape names without a profile
nonlinked_names <- scrape_data(url, 'p:nth-child(68) , p:nth-child(39) , p+ table td')
last_male_name_nonlink <- 'Water-Chaser'
last_female_name_nonlink <- 'Wonders-at-Stars'

nonlinked_names <- clean_strings(nonlinked_names)

# Add sex
# website has names ordered by sex, everyone before a certain name is male
unlinked <- tibble(name = nonlinked_names) %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_nonlink) ~ 'Male',
      id <= which(name == last_female_name_nonlink) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

linked <- as_tibble(linked_names) %>% 
  filter(!(name %in% c('[1]', '[2]', '[3]', '[4]', '[5]', '†', 'Argonian', 'Jel', 'Hist Sap')))

# Add sex
# website has names ordered by sex, everyone before a certain name is male
linked <- linked %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_link) ~ 'Male',
      id <= which(name == last_female_name_link) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

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
  filter(!(name %in% c('1', '2')))

argonians <- linked %>% bind_rows(unlinked) %>% mutate(race = 'Argonian') %>% distinct %>% 
  select(name, sex, race, url)



#### Bosmer Names ####
url <- 'https://en.uesp.net/wiki/Lore:Bosmer_Names'

# Scrape names with a link to their profile
linked_names <- scrape_all(url, 'p a')
last_male_name_link <- 'Vanirion'
last_female_name_link <- 'Willow'

# Scrape names without a profile
nonlinked_names <- scrape_data(url, 'p:nth-child(29) , table:nth-child(39) td , p:nth-child(62) , table:nth-child(73) td , table:nth-child(8) td')
last_male_name_nonlink <- 'Vicmond'
last_female_name_nonlink <- 'Yarmia'

nonlinked_names <- clean_strings(nonlinked_names)

# Add sex
# website has names ordered by sex, everyone before a certain name is male
unlinked <- tibble(name = nonlinked_names) %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_nonlink) ~ 'Male',
      id <= which(name == last_female_name_nonlink) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

linked <- as_tibble(linked_names) %>% 
  filter(!(name %in% c('Bosmer')))

# Add sex
# website has names ordered by sex, everyone before a certain name is male
linked <- linked %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_link) ~ 'Male',
      id <= which(name == last_female_name_link) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

# Bring over full name
# Some names were linked to stories about them, not their actual profile
# For these people we do not want to use their full name
# Some times two people had the same name but were different - these were pulled as a number (i.e., 1 and 2)
# This code will duplicate people -> give us their first name, then make another row with first name last name
linked <- linked %>% 
  mutate(full_name = str_remove(full_name, '( \\(.*\\))$')) %>% 
  filter(str_detect(full_name, name) | !is.na(as.numeric(name))) %>% 
  filter(name != full_name) %>% 
  mutate(name = full_name) %>% 
  bind_rows(linked) %>% 
  select(-full_name) %>%
  distinct %>% 
  filter(is.na(as.numeric(name))) # Remove any numeric names (only referenced when there were multiple same names)

# Create final list of names
bosmer <- linked %>% 
  bind_rows(unlinked) %>% 
  mutate(race = 'Bosmer') %>% 
  distinct(name, sex, race, url)


#### Breton Names ####
url <- 'https://en.uesp.net/wiki/Lore:Breton_Names'

# Scrape names with a link to their profile
# There are sooo many reused breton names in ESO!
linked_names <- scrape_all(url, 'h3+ p a , h2+ table a , p a+ a')
last_male_name_link <- 'Wynster'
last_female_name_link <- 'Ylanie'

# Scrape names without a profile
nonlinked_names <- scrape_data(url, 'h2+ table li , p:nth-child(78) , p:nth-child(52) , p:nth-child(27) , p+ table td')
last_male_name_nonlink <- 'Yves'
last_female_name_nonlink <- 'Yvonne'

nonlinked_names <- clean_strings(nonlinked_names)

# Add sex
# website has names ordered by sex, everyone before a certain name is male
unlinked <- tibble(name = nonlinked_names) %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_nonlink) ~ 'Male',
      id <= which(name == last_female_name_nonlink) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

linked <- as_tibble(linked_names) %>% 
  filter(!(name %in% c('Breton')))

# Add sex
# website has names ordered by sex, everyone before a certain name is male
linked <- linked %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_link) ~ 'Male',
      id <= which(name == last_female_name_link) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

# Bring over full name
# Some names were linked to stories about them, not their actual profile
# For these people we do not want to use their full name
# Some times two people had the same name but were different - these were pulled as a number (i.e., 1 and 2)
# This code will duplicate people -> give us their first name, then make another row with first name last name
linked <- linked %>% 
  mutate(full_name = str_remove(full_name, '( \\(.*\\))$')) %>% 
  filter(str_detect(full_name, name) | !is.na(as.numeric(name))) %>% 
  filter(name != full_name) %>% 
  mutate(name = full_name) %>% 
  bind_rows(linked) %>% 
  select(-full_name) %>%
  distinct %>% 
  filter(is.na(as.numeric(name))) # Remove any numeric names (only referenced when there were multiple same names)

# Create final list of names
breton <- linked %>% 
  bind_rows(unlinked) %>% 
  mutate(race = 'Breton') %>% 
  distinct(name, sex, race, url)

#### Deadra Names ####
url <- 'https://en.uesp.net/wiki/Lore:Daedra_Names'

# Scrape names with a link to their profile
linked_names <- scrape_all(url, 'dd a')

# There are no unlinked names on this page
# Sex is too difficult to add based on the layout of this page

linked <- as_tibble(linked_names) %>% 
  filter(!(name %in% c('Daedra')))

# Bring over full name
# Some names were linked to stories about them, not their actual profile
# For these people we do not want to use their full name
# Some times two people had the same name but were different - these were pulled as a number (i.e., 1 and 2)
# This code will duplicate people -> give us their first name, then make another row with first name last name
linked <- linked %>% 
  mutate(full_name = str_remove(full_name, '( \\(.*\\))$')) %>% 
  filter(str_detect(full_name, name) | !is.na(as.numeric(name))) %>% 
  filter(name != full_name) %>% 
  mutate(name = full_name) %>% 
  bind_rows(linked) %>% 
  select(-full_name) %>%
  distinct %>% 
  filter(is.na(as.numeric(name))) # Remove any numeric names (only referenced when there were multiple same names)

# Create final list of names
deadra <- linked %>% 
  mutate(race = 'Deadra', sex = NA) %>% 
  distinct(name, sex, race, url)


#### Dunmer Names ####
url <- 'https://en.uesp.net/wiki/Lore:Dunmer_Names'

# Scrape names with a link to their profile
linked_names <- scrape_all(url, 'h4+ p a , b+ a , b a , p a+ a , h3+ p a')
last_male_name_link <- 'Xiomara'
last_female_name_link <- 'Valyne'

# Scrape names without a profile
nonlinked_names <- scrape_data(url, 'table:nth-child(41) td , table:nth-child(74) td , p:nth-child(98) , p:nth-child(64) , p:nth-child(31) , table:nth-child(8) td')
last_male_name_nonlink <- 'Yeveth'
last_female_name_nonlink <- 'Voldsea'

nonlinked_names <- clean_strings(nonlinked_names)

# Add sex
# website has names ordered by sex, everyone before a certain name is male
unlinked <- tibble(name = nonlinked_names) %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_nonlink) ~ 'Male',
      id <= which(name == last_female_name_nonlink) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

linked <- as_tibble(linked_names) %>% 
  filter(!(name %in% c('Names', 'Dunmer')))

# Add sex
# website has names ordered by sex, everyone before a certain name is male
linked <- linked %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_link) ~ 'Male',
      id <= which(name == last_female_name_link) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

# Bring over full name
# Some names were linked to stories about them, not their actual profile
# For these people we do not want to use their full name
# Some times two people had the same name but were different - these were pulled as a number (i.e., 1 and 2)
# This code will duplicate people -> give us their first name, then make another row with first name last name
linked <- linked %>% 
  mutate(full_name = str_remove(full_name, '( \\(.*\\))$')) %>% 
  filter(str_detect(full_name, name) | !is.na(as.numeric(name))) %>% 
  filter(name != full_name) %>% 
  mutate(name = full_name) %>% 
  bind_rows(linked) %>% 
  select(-full_name) %>%
  distinct %>% 
  filter(is.na(as.numeric(name))) # Remove any numeric names (only referenced when there were multiple same names)

# Create final list of names
dunmer <- linked %>% 
  bind_rows(unlinked) %>% 
  mutate(race = 'Dunmer') %>% 
  distinct(name, sex, race, url)


#### Imperial Names ####
url <- 'https://en.uesp.net/wiki/Lore:Imperial_Names'

# Scrape names with a link to their profile
linked_names <- scrape_all(url, 'p a')
last_male_name_link <- 'Vitellus'
last_female_name_link <- 'Villea'

# Scrape names without a profile
nonlinked_names <- scrape_data(url, 'p:nth-child(64) , p:nth-child(44) , p:nth-child(24) , p:nth-child(6)')
last_male_name_nonlink <- 'Volusianus'
last_female_name_nonlink <- 'Zonara'

nonlinked_names <- clean_strings(nonlinked_names)

# Add sex
# website has names ordered by sex, everyone before a certain name is male
unlinked <- tibble(name = nonlinked_names) %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_nonlink) ~ 'Male',
      id <= which(name == last_female_name_nonlink) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

linked <- as_tibble(linked_names) %>% 
  filter(!(name %in% c('Imperial')))

# Add sex
# website has names ordered by sex, everyone before a certain name is male
linked <- linked %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_link) ~ 'Male',
      id <= which(name == last_female_name_link) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

# Bring over full name
# Some names were linked to stories about them, not their actual profile
# For these people we do not want to use their full name
# Some times two people had the same name but were different - these were pulled as a number (i.e., 1 and 2)
# This code will duplicate people -> give us their first name, then make another row with first name last name
linked <- linked %>% 
  mutate(full_name = str_remove(full_name, '( \\(.*\\))$')) %>% 
  filter(str_detect(full_name, name) | !is.na(as.numeric(name))) %>% 
  filter(name != full_name) %>% 
  mutate(name = full_name) %>% 
  bind_rows(linked) %>% 
  select(-full_name) %>%
  distinct %>% 
  filter(is.na(as.numeric(name))) %>%  # Remove any numeric names (only referenced when there were multiple same names)
  filter(name != 'Background History')

# Create final list of names
imperial <- linked %>% 
  bind_rows(unlinked) %>% 
  mutate(race = 'Imperial') %>% 
  distinct(name, sex, race, url)


#### Khajiit Names ####
url <- 'https://en.uesp.net/wiki/Lore:Khajiit_Names'

# Scrape names with a link to their profile
linked_names <- scrape_all(url, 'ul+ p a , h3+ p a')
last_male_name_link <- 'Eagle Eye'
last_female_name_link <- 'Sharp-Tongue'

# Scrape names without a profile
nonlinked_names <- scrape_data(url, 'p:nth-child(73) , p:nth-child(42) , ul~ p+ table td')
last_male_name_nonlink <- 'Zur'
last_female_name_nonlink <- 'Zurana'

nonlinked_names <- clean_strings(nonlinked_names)

# Add sex
# website has names ordered by sex, everyone before a certain name is male
unlinked <- tibble(name = nonlinked_names) %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_nonlink) ~ 'Male',
      id <= which(name == last_female_name_nonlink) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

# Add sex
# website has names ordered by sex, everyone before a certain name is male
linked <- as_tibble(linked_names) %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_link) ~ 'Male',
      id <= which(name == last_female_name_link) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id) %>% 
  mutate(url = ifelse(name == 'Turamane', 'https://en.uesp.net/wiki/Arena:Turamane_ap%27_Kolthis', url)) # manually add this persons link

# Bring over full name
# Some names were linked to stories about them, not their actual profile
# For these people we do not want to use their full name
# Some times two people had the same name but were different - these were pulled as a number (i.e., 1 and 2)
# This code will duplicate people -> give us their first name, then make another row with first name last name
linked <- linked %>% 
  mutate(full_name = str_remove(full_name, '( \\(.*\\))$')) %>% 
  filter(str_detect(full_name, name) | !is.na(as.numeric(name))) %>% 
  filter(name != full_name) %>% 
  mutate(name = full_name) %>% 
  bind_rows(linked) %>% 
  select(-full_name) %>%
  distinct %>% 
  filter(is.na(as.numeric(name)))  # Remove any numeric names (only referenced when there were multiple same names)

# Create final list of names
khajiit <- linked %>% 
  bind_rows(unlinked) %>% 
  mutate(race = 'Khajiit') %>% 
  distinct(name, sex, race, url)

# Note: multiple M'aiq the Liar - these are different people across different games - it is a running joke


#### Nord Names ####
url <- 'https://en.uesp.net/wiki/Lore:Nord_Names'

# Scrape names with a link to their profile
linked_names <- scrape_all(url, 'p a')
last_male_name_link <- 'Yust'
last_female_name_link <- 'Vunhilde'

# Scrape names without a profile
nonlinked_names <- scrape_data(url, 'table:nth-child(39) td , table:nth-child(69) td , p:nth-child(55) , p:nth-child(29) , table:nth-child(10) td')
last_male_name_nonlink <- 'Yngvar'
last_female_name_nonlink <- 'Yrna'

nonlinked_names <- clean_strings(nonlinked_names)

# Add sex
# website has names ordered by sex, everyone before a certain name is male
unlinked <- tibble(name = nonlinked_names) %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_nonlink) ~ 'Male',
      id <= which(name == last_female_name_nonlink) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

linked <- as_tibble(linked_names) %>% 
  filter(!(name %in% c('Nord')))

# Add sex
# website has names ordered by sex, everyone before a certain name is male
linked <- linked %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_link) ~ 'Male',
      id <= which(name == last_female_name_link) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

# Bring over full name
# Some names were linked to stories about them, not their actual profile
# For these people we do not want to use their full name
# Some times two people had the same name but were different - these were pulled as a number (i.e., 1 and 2)
# This code will duplicate people -> give us their first name, then make another row with first name last name
linked <- linked %>% 
  mutate(full_name = str_remove(full_name, '( \\(.*\\))$')) %>% 
  filter(str_detect(full_name, name) | !is.na(as.numeric(name))) %>% 
  filter(name != full_name) %>% 
  mutate(name = full_name) %>% 
  bind_rows(linked) %>% 
  select(-full_name) %>%
  distinct %>% 
  filter(is.na(as.numeric(name))) # Remove any numeric names (only referenced when there were multiple same names)

# Create final list of names
nord <- linked %>% 
  bind_rows(unlinked) %>% 
  mutate(race = 'Nord') %>% 
  distinct(name, sex, race, url)


#### Orc Names ####
url <- 'https://en.uesp.net/wiki/Lore:Orc_Names'

# Scrape names with a link to their profile
linked_names <- scrape_all(url, 'p a')
last_male_name_link <- 'Zbulgat'
last_female_name_link <- 'Zeg'

# Scrape names without a profile
nonlinked_names <- scrape_data(url, 'p:nth-child(79) , p:nth-child(43) , p:nth-child(23)')
last_male_name_nonlink <- 'Zumog'
last_female_name_nonlink <- 'Zubesha'

nonlinked_names <- clean_strings(nonlinked_names)

# Add sex
# website has names ordered by sex, everyone before a certain name is male
unlinked <- tibble(name = nonlinked_names) %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_nonlink) ~ 'Male',
      id <= which(name == last_female_name_nonlink) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

linked <- as_tibble(linked_names) %>% 
  filter(!(name %in% c('Orc')))

# Add sex
# website has names ordered by sex, everyone before a certain name is male
linked <- linked %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_link) ~ 'Male',
      id <= which(name == last_female_name_link) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

# Bring over full name
# Some names were linked to stories about them, not their actual profile
# For these people we do not want to use their full name
# Some times two people had the same name but were different - these were pulled as a number (i.e., 1 and 2)
# This code will duplicate people -> give us their first name, then make another row with first name last name
linked <- linked %>% 
  mutate(full_name = str_remove(full_name, '( \\(.*\\))$')) %>% 
  filter(str_detect(full_name, name) | !is.na(as.numeric(name))) %>% 
  filter(name != full_name) %>% 
  mutate(name = full_name) %>% 
  bind_rows(linked) %>% 
  select(-full_name) %>%
  distinct %>% 
  filter(is.na(as.numeric(name))) # Remove any numeric names (only referenced when there were multiple same names)

# Create final list of names
orc <- linked %>% 
  bind_rows(unlinked) %>% 
  mutate(race = 'Orc') %>% 
  distinct(name, sex, race, url) %>% 
  mutate(name = str_remove(name, 'A conversation with '))


#### Reachman Names ####
url <- 'https://en.uesp.net/wiki/Lore:Reachman_Names'

# Scrape names with a link to their profile
linked_names <- scrape_all(url, 'p a')
last_male_name_link <- 'Wunagh'
last_female_name_link <- 'Voanche'

# No non-linked NPCs

linked <- as_tibble(linked_names) %>% 
  filter(!(name %in% c('Reachmen')))

# Add sex
# website has names ordered by sex, everyone before a certain name is male
linked <- linked %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_link) ~ 'Male',
      id <= which(name == last_female_name_link) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

# Bring over full name
# Some names were linked to stories about them, not their actual profile
# For these people we do not want to use their full name
# Some times two people had the same name but were different - these were pulled as a number (i.e., 1 and 2)
# This code will duplicate people -> give us their first name, then make another row with first name last name
linked <- linked %>% 
  mutate(full_name = str_remove(full_name, '( \\(.*\\))$')) %>% 
  filter(str_detect(full_name, name) | !is.na(as.numeric(name))) %>% 
  filter(name != full_name) %>% 
  mutate(name = full_name) %>% 
  bind_rows(linked) %>% 
  select(-full_name) %>%
  distinct %>% 
  filter(is.na(as.numeric(name))) # Remove any numeric names (only referenced when there were multiple same names)

# Create final list of names
reachman <- linked %>% 
  mutate(race = 'Reachman') %>% 
  distinct(name, sex, race, url)


#### Redguard Names ####
url <- 'https://en.uesp.net/wiki/Lore:Redguard_Names'

# Scrape names with a link to their profile
linked_names <- scrape_all(url, 'h3+ p a')
last_male_name_link <- 'Zakhin'
last_female_name_link <- 'Zell'

# Scrape names without a profile
nonlinked_names <- scrape_data(url, 'p:nth-child(70) , p:nth-child(36)')
last_male_name_nonlink <- 'Ziyad'
last_female_name_nonlink <- 'Zayya'

nonlinked_names <- clean_strings(nonlinked_names)

# Add sex
# website has names ordered by sex, everyone before a certain name is male
unlinked <- tibble(name = nonlinked_names) %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_nonlink) ~ 'Male',
      id <= which(name == last_female_name_nonlink) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

# Add sex
# website has names ordered by sex, everyone before a certain name is male
linked <- as_tibble(linked_names)  %>% 
  mutate(id = row_number()) %>% 
  mutate(
    sex = case_when(
      id <= which(name == last_male_name_link) ~ 'Male',
      id <= which(name == last_female_name_link) ~ 'Female',
      TRUE ~ NA
    )
  ) %>% 
  select(-id)

# Bring over full name
# Some names were linked to stories about them, not their actual profile
# For these people we do not want to use their full name
# Some times two people had the same name but were different - these were pulled as a number (i.e., 1 and 2)
# This code will duplicate people -> give us their first name, then make another row with first name last name
linked <- linked %>% 
  mutate(full_name = str_remove(full_name, '( \\(.*\\))$')) %>% 
  filter(str_detect(full_name, name) | !is.na(as.numeric(name))) %>% 
  filter(name != full_name) %>% 
  mutate(name = full_name) %>% 
  bind_rows(linked) %>% 
  select(-full_name) %>%
  distinct %>% 
  filter(is.na(as.numeric(name))) # Remove any numeric names (only referenced when there were multiple same names)

# Create final list of names
redguard <- linked %>% 
  bind_rows(unlinked) %>% 
  mutate(race = 'Redguard') %>% 
  distinct(name, sex, race, url)


#### Combine ####

# Combine all scraped names together
npc_names <- bind_rows(
  altmer,
  argonians,
  bosmer,
  breton,
  deadra,
  dunmer,
  imperial,
  khajiit,
  nord,
  orc,
  reachman,
  redguard
)

#### Some more cleaning ####

# Some duplicates due to how the page was laid out
# These dups might have different values for sex or no - throw out the one that has sex missing
lore_names <- lore_names %>% 
  group_by(name, url) %>% 
  arrange(rowSums(is.na(.)), .by_group = T) %>%  # Arrange by ID and count of NAs in each row
  slice(1) %>% 
  filter(name != "") %>% 
  ungroup

# No names will start with the
lore_names <- lore_names %>% filter(!str_detect(tolower(name), '^the '))

# Remove some non-names
remove <- lore_names %>% 
  filter(str_detect(url, 'Lore|Tribunal')) %>% 
  filter(str_detect(name, "among the|of the|from the|A ")) %>% 
  filter(!(name %in% c('Jeek of the River', 'Hagrof the Righteous'))) %>% 
  pull(name)

lore_names <- lore_names %>% filter(!(name %in% remove))

write_csv(lore_names, 'data/lore_names.csv')
