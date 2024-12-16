library(tidyverse)

TES_names <- read_csv('~/elder-scrolls-name-generation/data/archive/TES_names_uncleaned.csv')

TES_names <- TES_names %>% filter(name != "")

# Remove the prefix Jarl in all "Jarl [Name]" for Skyrim characters
# The URL for non Skyrim (ESO) Jarls has their Jarl title in it, Skyrim does not

# Same thing for Count|Countess, except most of the time the URL has "Count", sometimes it does not
# Manually checked all of these and hardcoded it in
TES_names <- TES_names %>% mutate(
  name = case_when(
    game == 'Skyrim' ~ str_replace_all(name, "^Jarl\\s+", ""),
    name %in% c(
      'Corvus Umbranox', 'Ormellius Goldwine', 'Janus Hassildor'
    ) ~ str_replace_all(name, "^Countess\\s+|^Count\\s+", ""),
    name == 'Grandmaster Jauffre' ~ 'Jauffre',
    name == 'The Archmagister' ~ 'Archmagister',
    name == "Turamane Ap'Kolthis" ~ "Turamane ap'Kolthis",
    TRUE ~ name
  ),
  game = case_when(
    name %in% c('Karliah', 'Camaron', 'Selene') ~ 'Lore',
    TRUE ~ game
  )
) 

# Create a link to each NPC's online profile
TES_names <- TES_names %>% 
  mutate(
    url = paste0(
      'https://en.uesp.net/wiki/', 
      game, ':',
      str_replace_all(name, ' ', '_')
    )
  )

# Clean some URLs manually
TES_names <- TES_names %>% 
  mutate(
    url = case_when(
      name == "Turamane ap'Kolthis" ~ str_replace(url, "'", '%27_'),
      TRUE ~ url
    )
  )

# Now, after URL has been created, go back and remove Jarl, Count, Countess title from all people
TES_names <- TES_names %>% mutate(
  name = str_replace_all(name, "^Jarl\\s+", ""),
  name = str_replace_all(name, "^Countess\\s+|^Count\\s+", ""),
  name = str_replace_all(name, "^Grandmaster\\s+", "")
) %>% 
  group_by(name, game) %>% 
  filter(row_number() == 1) %>% 
  ungroup

# Get rid of easily detectable unwanted names
guards <- TES_names$name[str_detect(tolower(TES_names$name), tolower('Guards'))]
initiate <- TES_names$name[str_detect(tolower(TES_names$name), tolower('.* Initiate'))]
dlc <- TES_names %>% filter(name %in% c('CC', 'HF', 'DG', 'DB')) %>% pull(name)
footnotes <- c('[a]', '[c]', '[d]') # more hardcoding

TES_names <- TES_names %>% filter(!(name %in% c(guards, initiate, dlc, footnotes)))

write_csv(TES_names, '~/elder-scrolls-name-generation/data/TES_names.csv')
