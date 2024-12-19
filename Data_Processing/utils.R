scrape_url <- function(url, extract) {
  tryCatch({
    # Read the HTML content of the page
    page <- read_html(url, extract)
    
    # Extract the href attribute from links (anchor tags)
    links <- page %>% html_nodes(extract) %>% html_attr("href")
    
    # Remove NULL or NA values and return
    links <- links[!is.na(links)]
    return(links)
  }, error = function(e) {
    message(paste("Failed to scrape:", url))
    return(NULL)
  })
}

scrape_data <- function(url, extract, sleep = FALSE) {
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

scrape_all <- function(url, extract){
  output <- list()
  output[['name']] <- scrape_data(url, extract)
  
  cleaned_url <- paste0('https://en.uesp.net',scrape_url(url, extract))
  output[['url']] <- str_replace(cleaned_url, "#.*$", "") # remove any link to somewhere later in the page
  
  full_name <- str_replace_all(str_extract(output$url, "(?<=:)[^/]+$"), '_', ' ')
  full_name <- str_replace_all(full_name, '%27', "'")
  full_name <- str_replace_all(full_name, '%22', '') # This is " but it makes the data look weird so I'm just removing it
  output[['full_name']] <- full_name
  
  return(output)
}

clean_strings <- function(vec) {
  # Remove trailing commas
  cleaned <- str_remove(vec, ",$")
  
  # Remove custom words
  cleaned <- str_replace_all(cleaned, 'Single-word:\n|Hyphenated:\n|Tamrielic:\n', ', ')
  
  # Split strings with ", " into individual elements
  split_elements <- str_split(cleaned, ", ")
  
  # Flatten the list and extract unique elements
  unique_elements <- unique(unlist(split_elements))
  
  # Remove parentheses and their contents
  cleaned_strings <- str_replace_all(unique_elements, "\\s*\\(.*?\\)", "")
  
  return(cleaned_strings)
}