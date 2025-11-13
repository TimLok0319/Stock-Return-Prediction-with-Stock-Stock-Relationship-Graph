import os
import json
from pathlib import Path
from tqdm import tqdm  # For a nice progress bar
import logging

""
"This file is use to extract targeted stock's news from the RAW DATA"
""

# --- Configuration ---

# 1. Define the 10 target stocks your GNN will model.
#    The script will check for co-mentions among these names.
TARGET_STOCKS = [
   "apple","microsoft","google","amazon","nvidia","jpmorgan","berkshire hathaway","johnson & johnson", "walmart", "exxon"
]
# Create a fast-lookup set for efficient checking
TARGET_STOCK_SET = set(TARGET_STOCKS)

# 2. Define the input folder where your raw JSONs are stored.
INPUT_FOLDER = Path("./rawData/2018May")

# 3. --- DEBUG MODE ---
#    Set to True to print results to the terminal (as you asked).
#    Set to False to save to the 'OUTPUT_FILE' for your real pipeline.
DEBUG_MODE = False
DEBUG_FILE_LIMIT = 500  # How many files to check in debug mode
OUTPUT_FILE = Path("./cleaned_high_signal_articles.jsonl")

# --- End of Configuration ---


# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def find_json_files(folder_path):
    """Finds all .json files in a directory, recursively."""
    json_files = []
    log.info(f"Scanning for .json files in {folder_path}...")
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(Path(root) / file)
    log.info(f"Found {len(json_files)} .json files.")
    return json_files

def extract_needed_data(file_path):
    """
    Opens a single JSON file, extracts the needed keys based on the
    new structure, and checks for 2+ co-mentions.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. Extract the key-value pairs you requested
        uuid = data.get('uuid')
        published_date = data.get('published')
        title = data.get('title')
        text = data.get('text')
        entities_obj = data.get('entities')

        # 2. Check for basic data integrity
        if not (uuid and published_date and title and text and entities_obj):
            return None  # Skip file if essential data is missing

        # 3. This is the "Co-Mention" filter (Your "Step 2")
        #    This logic is NEW to match your sample JSON structure.
        
        # Get the list of organization objects
        organization_list = entities_obj.get('organizations', [])
        
        found_targets = []
        for org_object in organization_list:
            # The 'name' is inside an object like: {"name": "reuters", "sentiment": "negative"}
            org_name = org_object.get('name')
            
            if org_name and (org_name in TARGET_STOCK_SET):
                found_targets.append(org_name)
        
        # Get unique mentions
        unique_targets = set(found_targets)

        # 4. Apply the filter: We ONLY care about articles with 2+ co-mentions
        if len(unique_targets) < 1:
            return None  # This is a "noise" article. DISCARD.
        
        # 5. This is a "High-Signal" article. Package the data for the LLM pipeline.
        clean_data = {
            "uuid": uuid,
            "published": published_date,
            "title": title,
            "text": text,
            "found_entities": list(unique_targets) # Save the list of companies we found
        }
        
        return clean_data

    except json.JSONDecodeError:
        log.warning(f"Skipping corrupt JSON file: {file_path}")
        return None
    except Exception as e:
        log.error(f"An unknown error occurred with {file_path}: {e}")
        return None

def main():
    """
    Main pipeline function.
    """
    if not INPUT_FOLDER.exists():
        log.error(f"Input folder not found: {INPUT_FOLDER}")
        log.error("Please create this folder and put your raw JSON files inside it.")
        return

    json_files = find_json_files(INPUT_FOLDER)
    
    if not json_files:
        log.warning("No .json files found. Exiting.")
        return

    # --- THIS IS THE MODIFIED SECTION ---

    if DEBUG_MODE:
        log.info(f"--- RUNNING IN DEBUG MODE (PRINTING TO TERMINAL) ---")
        log.info(f"Checking the first {DEBUG_FILE_LIMIT} files...")
        
        articles_saved = 0
        
        # We slice the list here for a limited test run
        files_to_check = json_files[:min(DEBUG_FILE_LIMIT, len(json_files))]

        for file_path in tqdm(files_to_check, desc="Debug Filtering"):
            clean_data = extract_needed_data(file_path)
            
            if clean_data:
                # If the data is good, print it to the terminal
                print("\n--- --------------------------- ---")
                print(f"--- FOUND HIGH-SIGNAL ARTICLE ---")
                print(f"--- File: {file_path.name}")
                print(f"--- --------------------------- ---")
                print(f"UUID: {clean_data['uuid']}")
                print(f"Published: {clean_data['published']}")
                print(f"Title: {clean_data['title']}")
                print(f"Found Entities: {clean_data['found_entities']}")
                print("---------------------------------\n")
                articles_saved += 1
        
        log.info(f"--- Debug Run Complete ---")
        log.info(f"Processed {len(files_to_check)} files.")
        log.info(f"Found {articles_saved} high-signal (co-mention) articles.")

    else:
        # This is the "production" code to write the file
        log.info(f"--- RUNNING IN PRODUCTION MODE (WRITING TO FILE) ---")
        log.info(f"Starting extraction... Output will be saved to {OUTPUT_FILE}")

        articles_saved = 0
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
            for file_path in tqdm(json_files, desc="Filtering articles"):
                
                clean_data = extract_needed_data(file_path)
                
                if clean_data:
                    # Write it as a new line in the output file
                    json.dump(clean_data, out_f)
                    out_f.write('\n')
                    articles_saved += 1
        
        log.info("--- Pipeline Complete ---")
        log.info(f"Processed {len(json_files)} total raw files.")
        log.info(f"Found and saved {articles_saved} high-signal (co-mention) articles.")
        log.info(f"Your clean data is ready in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()