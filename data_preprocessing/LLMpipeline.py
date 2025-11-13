from google import genai
from google.genai import types
from itertools import combinations
from tqdm import tqdm
import json

with open('API_KEY.json', 'r') as file:
    config = json.load(file)

GEMINI_API_KEY = config['key']
client = genai.Client(api_key=GEMINI_API_KEY)
all_llm_outputs = []
total_output_tokens = 0  # <-- ADD THIS
total_input_tokens = 0

# Pipeline A: Single-Entity (Diagonal)
# Create the new Client object ONCE

# --- 2. Define Your Two Pipelines (Prompts & Schemas) ---

# Pipeline A: Single-Entity (Diagonal)
single_entity_system_prompt = """You are a financial analyst. Your task is to read a news article and determine the direct financial sentiment for a *single* specified company.
Focus only on the impact of the news on the company. Respond ONLY with a JSON object."""

# Define the schema as a Python-typed dictionary
class SingleEntityOutput(types.TypedDict):
    published_date: str
    sentiment_score: float
    sentiment_type: str
    reasoning: str

# Create the GenerationConfig object
single_entity_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=SingleEntityOutput
)

# Pipeline B: Relationship (Off-Diagonal)
relationship_system_prompt = """You are a financial analyst. Your task is to read a news article and find the single, most important semantic relationship between two specified companies.
Respond ONLY with a JSON object."""

# Define the schema as a Python-typed dictionary
class RelationshipOutput(types.TypedDict):
    published_date: str
    relationship_type: str
    sentiment_score: float

# Create the GenerationConfig object
relationship_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=RelationshipOutput
)

# --- 3. Load Your JSON File ---
try:
    with open('testingJson.json', 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
except FileNotFoundError:
    print("Error: Input file not found.")
    exit()

print(f"Loaded {len(articles_data)} articles from file.")

# --- 4. Loop Through Articles and Call API ---
all_llm_outputs = []
articles_to_process = articles_data

print(f"--- Processing {len(articles_to_process)} articles... ---")

for article in tqdm(articles_to_process, desc="Processing articles"):
    
    article_text = article.get('text')
    entities = article.get('found_entities', [])
    date = article.get('published', 'UNKNOWN_DATE')[:10]

    try:
        # === POSSIBILITY 1: SINGLE-ENTITY ===
        if len(entities) == 1:
            company = entities[0]
            user_prompt = f"Article: \"{article_text}\"\nDate: \"{date}\"\nCompany: \"{company}\""
            
            # Use the new client.models.generate_content syntax
            response = client.models.generate_content(
                model="gemini-2.5-flash", # Specify model here
                contents=[single_entity_system_prompt, user_prompt],
                config=single_entity_config
            )
            total_output_tokens += response.usage_metadata.candidates_token_count  # <-- ADD THIS
            total_input_tokens += response.usage_metadata.prompt_token_count
            result = json.loads(response.text)
            result['processed_company'] = company 
            all_llm_outputs.append(result)

        # === POSSIBILITY 2: RELATIONSHIP ===
        elif len(entities) >= 2:
            unique_entities = sorted(list(set(entities)))
            for pair in combinations(unique_entities, 2):
                company_a, company_b = pair
                user_prompt = f"Article: \"{article_text}\"\nDate: \"{date}\"\nCompany A: \"{company_a}\"\nCompany B: \"{company_b}\""
                
                # Use the new client.models.generate_content syntax
                response = client.models.generate_content(
                    model="gemini-2.5-flash", # Specify model here
                    contents=[relationship_system_prompt, user_prompt],
                    config=relationship_config
                )
                total_output_tokens += response.usage_metadata.candidates_token_count  # <-- ADD THIS
                total_input_tokens += response.usage_metadata.prompt_token_count
                result = json.loads(response.text)
                result['processed_pair'] = [company_a, company_b]
                all_llm_outputs.append(result)
        
        # === POSSIBILITY 3: SKIPPED ===
        else:
            pass 

    except Exception as e:
        print(f"\nError processing article (UUID {article.get('uuid')}): {e}")
        pass

# --- 5. Save Your Results ---
output_filename = 'llm_results.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(all_llm_outputs, f, indent=4)

print(f"\n--- DONE ---")
print(f"Successfully processed articles and saved {len(all_llm_outputs)} results to '{output_filename}'.")
print(f"Total output tokens consumed: {total_output_tokens}")  # <-- ADD THIS
print(f"Total input tokens consumed: {total_input_tokens}")  # <-- ADD THIS