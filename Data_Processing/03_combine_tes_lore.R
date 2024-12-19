library(tidyverse)

# Found some names from TES_names that were not in lore_names
# For example, Rune from the thieves guild in skyrim
# Add these names - I manually looked through and filtered out some people who were not actual people
tes_names <- read_csv('data/TES_names.csv')
lore_names <- read_csv('data/lore_names.csv')

combine <- tes_names %>% anti_join(lore_names, by = c('name')) %>%
  filter(
    !(str_detect(name, "Night Mother| Guard| Soldier|Empire| Agent| Captain|Priestess| Orc|\\[|Prophet|Pale Lady|Dark Minion|Prisoner|The Squid|Figure|Statue|Sailor|Debtor|Combatant|Fisherman|
Direnni|Spirit|Worker|Northpoint|Alik'r|^The |Stendarr|Body|Refugee|Pact|Nord|Herald|Boat|Merchant|Redoran|Monster|Conjured|Apparition|Camp|Hunter|Housecarl|Projection|	
Memory| of |Imperial|Rebel|Headsman|Adventurer|Watchman|Vampire|Miners|Sacrifice|Troll|Madman|Wizard|Stranger|Fan|Shady|Monkey"))
  )

npc_names <- lore_names %>% 
  bind_rows(combine) %>% 
  arrange(name) %>% 
  select(-game)

# Drop all NA values unless all values for that id are NA, then just keep one row
npc_names <- npc_names %>% group_by(name, sex, race) %>% 
  filter(!all(is.na(url)) | row_number() == 1) %>%
  filter(!is.na(url) | all(is.na(url))) %>% ungroup

npc_names <- npc_names %>% filter(
  !(name == '†' | str_detect(name, '^[a-z]|\\['))
)

write_csv(npc_names, 'npc_data.csv')
# write_delim(as.data.frame(unique(npc_names$name)), 'npc_names.txt', delim = '\n', col_names = F)
