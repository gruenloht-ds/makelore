library(tidyverse)
library(rvest)

# Web scrape character names from the elder scrolls (TES) games
# https://en.uesp.net/wiki/Main_Page

#### SKYRIM ####
url <- "https://en.uesp.net/wiki/Skyrim:People"
html_page <- read_html(url)

# Note dragons and other creatures are not scraped in this search

# This match will miss people from random encounters as these people are formatted in a list "li b a"
table_cells <- html_page %>%
  html_nodes('td > b a') %>%
  html_text()

# Check for people scraped
table_cells[str_detect(tolower(table_cells), 'serana')]
table_cells[str_detect(tolower(table_cells), 'harkon')]
table_cells[str_detect(tolower(table_cells), 'ebony')]
table_cells[str_detect(tolower(table_cells), "j'zargo")]
table_cells[str_detect(tolower(table_cells), "'")]

# Match for people who occur during random encounters
random_encounter <- html_page %>% 
  html_nodes('li b a') %>% 
  html_text()

skyrim_people <- tibble(name = c(table_cells, random_encounter))

# Remove duplicates (people might have lived in multiple different places resulting in multiple mentions)
skyrim_people <- skyrim_people %>% 
  group_by(name) %>% 
  filter(row_number() == 1) %>% 
  ungroup

print(paste0('There are a total of ', nrow(skyrim_people), ' Skyrim characters'))
# "There are a total of 1031 Skyrim characters"


#### OBLIVION ####
url <- "https://en.uesp.net/wiki/Oblivion:People"
html_page <- read_html(url)

# This match will miss people who do not physically show up in the game
# This will be fine as some of those names are joke names
# Only miss out on an additional 40 potential names
table_cells <- html_page %>%
  html_nodes(".wikitable dt a , b a") %>%
  html_text()

oblivion_people <- tibble(name = c(table_cells))

# Remove duplicates (people might have lived in multiple different places resulting in multiple mentions)
oblivion_people <- oblivion_people %>% 
  group_by(name) %>% 
  filter(row_number() == 1) %>% 
  ungroup

print(paste0('There are a total of ', nrow(oblivion_people), ' Oblivion characters'))
# "There are a total of 921 Oblivion characters"

#### MORROWIND ####

# The morrowind layout is different. There are subpages the contain lists of npc names
# Create a function to cycle through these pages since the layout in each subpage is the same
pull_names <- function(url, extract = 'b a') {
  result <- tryCatch({
    html_page <- read_html(url)
    
    table_cells <- html_page %>%
      html_nodes(extract) %>%
      html_text()
  }, error = function(e) {
    return(NULL)  
  })
  
  return(result)
}

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

tmp <- unlist(sapply(list_of_urls, pull_names), use.names = F)
morrowind_people <- tibble(name = tmp)
morrowind_people <- morrowind_people %>% 
  group_by(name) %>% 
  filter(row_number() == 1) %>% 
  ungroup


#### DAGGERFALL ####
url <- "https://en.uesp.net/wiki/Daggerfall:People"
html_page <- read_html(url)

table_cells <- html_page %>%
  html_nodes(".wikitable a") %>%
  html_text()

daggerfall_people <- tibble(name = table_cells)
daggerfall_people <- daggerfall_people %>% 
  group_by(name) %>% 
  filter(row_number() == 1) %>% 
  ungroup

#### ARENA ####

url <- "https://en.uesp.net/wiki/Arena:People"
html_page <- read_html(url)

table_cells <- html_page %>%
  html_nodes(".mw-headline") %>%
  html_text()

arena_people <- tibble(name = table_cells)
arena_people <- arena_people %>% 
  group_by(name) %>% 
  filter(row_number() == 1) %>% 
  ungroup

#### ELDER SCROLLS ONLINE ####

# This page has many subpages containing lists of people
url <- "https://en.uesp.net/wiki/Online:People"
html_page <- read_html(url)

# Scrape for the names of those pages
section_headers <- html_page %>%
  html_nodes("h3+ ul a , h4+ ul a") %>%
  html_text()

# Create URLs from the page names
section_urls <- paste0('https://en.uesp.net/wiki/Online:', str_replace_all(section_headers, ' ', '_'))

# Extract names from each subpage
tmp <- unlist(sapply(section_urls, pull_names, extract = '.wikitable b a'), use.names = F)

eso_people <- tibble(name = tmp)
eso_people <- eso_people %>% 
  group_by(name) %>% 
  filter(row_number() == 1) %>% 
  ungroup

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
  filter(name != "")  %>% 
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

# Fun analysis of same names that were used in two different games
TES_names %>% 
  group_by(name) %>% 
  filter(n() > 1) %>%
  ungroup %>% 
  arrange(name) 

# Create a link to each NPC's online profile
TES_names <- TES_names %>% 
  mutate(
    url = paste0(
      'https://en.uesp.net/wiki/', 
      game, ':',
      str_replace_all(name, ' ', '_')
    )
  )

write_csv(TES_names, '~/elder-scrolls-name-generation/data/TES_names.csv')
