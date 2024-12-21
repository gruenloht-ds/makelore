library(tidyverse)
library(rvest)

# Web scrape character names from the elder scrolls (TES) games
# https://en.uesp.net/wiki/Main_Page

# Create function to extract data from a URL
extract_data <- function(url, extract, sleep = FALSE) {
  result <- tryCatch({
    html_page <- read_html(url)
    
    table_cells <- html_page %>%
      html_nodes(extract) %>%
      html_text()
  }, error = function(e) {
    return(NULL)  
  })
  
  if (sleep){
    Sys.sleep(runif(1, 1, 3)) # Randomized delay between 1 and 3 seconds to avoid overloading website
  }
  
  return(result)
}

#### SKYRIM ####
url <- "https://en.uesp.net/wiki/Skyrim:People"

# Note dragons and other creatures are not scraped in this search
# This match will miss people from random encounters as these people are formatted in a list
table_cells <- extract_data(url, 'td > b a')

# Match for people who occur during random encounters
random_encounter <- extract_data(url, 'li b a')

# Remove duplicates (people might have lived in multiple different places resulting in multiple mentions)
skyrim_people <- tibble(name = c(table_cells, random_encounter)) %>% 
  group_by(name) %>% 
  filter(row_number() == 1) %>% 
  ungroup


#### OBLIVION ####
url <- "https://en.uesp.net/wiki/Oblivion:People"

# This match will miss people who do not physically show up in the game
# This will be fine as some of those names are joke names
# Only miss out on an additional 40 potential names
table_cells <- extract_data(url, '.wikitable dt a , b a')

# Remove duplicates (people might have lived in multiple different places resulting in multiple mentions)
oblivion_people <- tibble(name = c(table_cells)) %>% 
  group_by(name) %>% 
  filter(row_number() == 1) %>% 
  ungroup


#### MORROWIND ####

# The morrowind layout is different. There are subpages the contain lists of npc names
# Cycle through these pages since the layout in each subpage is the same
list_of_urls <- c(
  'https://en.uesp.net/wiki/Morrowind:City_People',
  'https://en.uesp.net/wiki/Morrowind:Town_People',
  'https://en.uesp.net/wiki/Morrowind:Legion_Fort_People',
  'https://en.uesp.net/wiki/Morrowind:Ashlander_Camp_People',
  'https://en.uesp.net/wiki/Morrowind:Daedric_Ruins_People',
  'https://en.uesp.net/wiki/Morrowind:Dunmer_Stronghold_People',
  'https://en.uesp.net/wiki/Morrowind:Dwemer_Ruins_People',
  'https://en.uesp.net/wiki/Morrowind:Vampire_Stronghold_People',
  'https://en.uesp.net/wiki/Morrowind:Cave_People',
  'https://en.uesp.net/wiki/Morrowind:Mine_People',
  'https://en.uesp.net/wiki/Morrowind:Stronghold_People',
  'https://en.uesp.net/wiki/Morrowind:Tower_People',
  'https://en.uesp.net/wiki/Morrowind:Tomb_People',
  'https://en.uesp.net/wiki/Morrowind:Farm_People',
  'https://en.uesp.net/wiki/Morrowind:Other_People',
  'https://en.uesp.net/wiki/Morrowind:Wilderness_People'
)

# This does miss some NPCs, but does get most of them
# For example, there are a few here https://en.uesp.net/wiki/Morrowind:People_in_Balmora that are missed
tmp <- unlist(sapply(list_of_urls, extract_data, extract = 'b a', sleep = TRUE), use.names = F)

morrowind_people <- tibble(name = tmp) %>% 
  group_by(name) %>% 
  filter(row_number() == 1) %>% 
  ungroup


#### DAGGERFALL ####
url <- "https://en.uesp.net/wiki/Daggerfall:People"

table_cells <- extract_data(url, '.wikitable a')

daggerfall_people <- tibble(name = table_cells) %>% 
  group_by(name) %>% 
  filter(row_number() == 1) %>% 
  ungroup

#### ARENA ####

url <- "https://en.uesp.net/wiki/Arena:People"

table_cells <- extract_data(url, '.mw-headline')

arena_people <- tibble(name = table_cells) %>% 
  group_by(name) %>% 
  filter(row_number() == 1) %>% 
  ungroup

#### ELDER SCROLLS ONLINE ####

# This page has many subpages containing lists of people
url <- "https://en.uesp.net/wiki/Online:People"

# Scrape for the names of those pages
section_headers <- extract_data(url, "h3+ ul a , h4+ ul a")

# Create URLs from the page names
section_urls <- paste0('https://en.uesp.net/wiki/Online:', str_replace_all(section_headers, ' ', '_'))

# Extract names from each subpage
tmp <- unlist(sapply(section_urls, extract_data, extract = '.wikitable b a', sleep = FALSE), use.names = F) # For some reason when using sleep some webpages fail, but is fine when sleep is not used. idk why this is happening

eso_people <- tibble(name = tmp) %>% 
  group_by(name) %>% 
  filter(row_number() == 1) %>% 
  ungroup


#### COMBINE ####

# Combine all Elder Scrolls NPC names
TES_names <- bind_rows(
  skyrim_people,
  oblivion_people,
  morrowind_people,
  daggerfall_people,
  arena_people,
  eso_people,
  .id = 'game'
) %>% 
  mutate(
    game = case_when(
      game == '1' ~ 'Skyrim',
      game == '2' ~ 'Oblivion',
      game == '3' ~ 'Morrowind',
      game == '4' ~ 'Daggerfall',
      game == '5' ~ 'Arena',
      .default = 'Online'
    )
  ) %>% 
  select(name, game)

#### Some Cleaning ####

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

write_csv(TES_names, 'data/TES_names.csv')
