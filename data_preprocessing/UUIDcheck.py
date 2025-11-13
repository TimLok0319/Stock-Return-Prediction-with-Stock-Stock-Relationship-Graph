import json
import os

def load_uuids_from_jsonl(filepath: str) -> set:
    """Reads a .jsonl file line by line and returns a set of all UUIDs."""
    uuids = set()
    print(f"Loading UUIDs from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    uuid = data.get('uuid')
                    if uuid:
                        uuids.add(uuid)
                except json.JSONDecodeError:
                    print(f"  Warning: Skipping malformed JSON on line {i+1}")
    except FileNotFoundError:
        print(f"  Error: File not found: {filepath}")
    return uuids

def load_uuids_from_json(filepath: str) -> set:
    """Reads a .json file (a list of objects) and returns a set of all UUIDs."""
    uuids = set()
    print(f"Loading UUIDs from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            if not isinstance(data_list, list):
                print(f"  Error: {filepath} is not a JSON list.")
                return uuids
                
            for item in data_list:
                uuid = item.get('uuid')
                if uuid:
                    uuids.add(uuid)
    except FileNotFoundError:
        print(f"  Error: File not found: {filepath}")
    except json.JSONDecodeError:
        print(f"  Error: Could not decode {filepath}.")
    return uuids

def main():
    # --- CONFIGURE YOUR FILENAMES HERE ---
    # This is your original .jsonl file (the source of truth)
    JSONL_FILE = 'cleaned_high_signal_articles.jsonl' 
    
    # This is the final .json file your script converted it into
    JSON_FILE = 'cleaned_high_signal_articles.json' 
    # ---
    
    # 1. Load both sets of UUIDs
    jsonl_uuids = load_uuids_from_jsonl(JSONL_FILE)
    json_uuids = load_uuids_from_json(JSON_FILE)

    if not jsonl_uuids or not json_uuids:
        print("\nCould not perform comparison due to file errors.")
        return

    # 2. Perform set difference
    # This finds items in json_uuids that are NOT in jsonl_uuids
    uuids_only_in_json = json_uuids - jsonl_uuids
    
    # This finds items in jsonl_uuids that are NOT in json_uuids
    uuids_only_in_jsonl = jsonl_uuids - json_uuids

    # 3. Report the findings
    print("\n--- UUID Discrepancy Report ---")
    print(f"Total UUIDs in {JSONL_FILE}:\t {len(jsonl_uuids)}")
    print(f"Total UUIDs in {JSON_FILE}:\t {len(json_uuids)}")
    print("-" * 30)
    
    if not uuids_only_in_json:
        print("✅ SUCCESS: The files are identical.")
        print(f"  All UUIDs from {JSON_FILE} are present in {JSONL_FILE}.")
    else:
        print(f"🔥 DISCREPANCY FOUND: {len(uuids_only_in_json)} UUIDs exist in {JSON_FILE} but NOT in {JSONL_FILE}.")
        print("  These 'ghost' UUIDs are:")
        for i, uuid in enumerate(uuids_only_in_json):
            if i > 50: # Limit printout
                print(f"    ...and {len(uuids_only_in_json) - i} more.")
                break
            print(f"    - {uuid}")
            
    if uuids_only_in_jsonl:
        print(f"\nWarning: {len(uuids_only_in_jsonl)} UUIDs exist in {JSONL_FILE} but were missing from {JSON_FILE}.")


if __name__ == "__main__":
    main()