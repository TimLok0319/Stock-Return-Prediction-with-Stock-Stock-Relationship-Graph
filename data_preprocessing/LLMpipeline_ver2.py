import os
import json
from itertools import combinations
from tqdm import tqdm
from google import genai
from google.genai import types

with open('API_KEY.json', 'r') as file:
    config = json.load(file)

GEMINI_API_KEY = config['key']
client = genai.Client(api_key=GEMINI_API_KEY)

# --- 2. Define Your Two Pipelines (Prompts & Schemas) ---

# Pipeline A: Single-Entity (Diagonal)
single_entity_system_prompt = """You are a financial analyst. Your task is to read a news article and determine the direct financial sentiment for a single specified company.

Your goal is to respond ONLY with a JSON object containing three keys:

1.  **`material_event_type`**: A single category for the event.
    * Examples: 'EarningsReport', 'ProductLaunch', 'LegalTrouble', 'ExecutiveChange', 'AnalystRating', 'MarketNoise', 'NoEvent'

2.  **`sentiment_score`**: A qualitative score from -1.0 to +1.0 based on the semantic context of the event and its severity.

3.  **`reasoning`**: A brief, one-sentence explanation for your score and type (e.g., "Assigning +0.7 for 'EarningsReport' as revenue smashed expectations.").

**Scoring Guidelines:**
* You **must** distinguish between financially material events and non-material noise.
* Assign `0.0` with type 'MarketNoise' or 'NoEvent' for non-financial noise (e.g., CEO's personal life, political opinions, or generic market reports that just mention the company).
* Focus only on the score for events that clearly impact the company's revenue, costs, strategy, or risk.
+1.0 (Transformative Win): "Johnson & Johnson's new drug is approved and hailed as a cure for a major disease."
+0.9 (Major R&D Breakthrough): "NVIDIA announces a successful Phase 3 trial for a new blockbuster drug."
+0.8 (Major Product Success): "NVIDIA's new GPU launch sees record-breaking pre-orders and is 10x faster than the competition."
+0.7 (Strong Earnings / Strategy): "Apple smashes quarterly earnings expectations; e-commerce sales jump 40%."
+0.6 (Strategic Acquisition): "Apple acquired Buddybuild, a strategic move to bolster its software ecosystem."
+0.5 (Good Earnings / New Contract): "Google beats earnings, but by a smaller margin."
+0.4 (Positive Analyst / Upgrade): "A major analyst at Goldman Sachs upgrades Apple from 'Hold' to 'Strong Buy'."
+0.3 (Standard Business Update): "NVIDIA announces a standard, next-generation product refresh."
+0.2 (Slightly Positive): "A market report mentions Google as a stable performer."
+0.1 (Barely Positive): Very weak positive signal.
0.0 (Neutral / Noise): "Exxon shareholders vote to not split the CEO/chair roles." (This is a non-event, a vote for the status quo)
-0.1 (Barely Negative): Very weak negative signal.
-0.2 (Slightly Negative / Downgrade): "Analysts are 'cautious' on Walmart."
-0.3 (Minor Negative / Supply Chain): "A new report highlights minor supply chain concerns for Apple."
-0.4 (Vague Concerns): "Google's new product is seen as a 'potential' miss."
-0.5 (Missed Earnings / Minor Setback): "Google misses quarterly earnings expectations."
-0.6 (Negative Management / Strategy): "NVIDIA's long-time, visionary CEO suddenly resigns."
-0.7 (Major Lawsuit / Product Failure): "A significant patent infringement lawsuit is filed against NVIDIA."
-0.8 (Serious Setback / Product Recall): "Google's new phone is recalled due to a major defect."
-0.9 (Severe Event / R&D Failure): "Johnson & Johnson's drug trial is halted by the FDA for safety concerns."
-1.0 (Catastrophic / Fraud): "A federal investigation for fraud is opened on Apple's executive team."

Respond ONLY with a JSON object."""

class SingleEntityOutput(types.TypedDict):
    published_date: str
    sentiment_score: float
    material_event_type: str
    reasoning: str

single_entity_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=SingleEntityOutput
)

# Pipeline B: Relationship (Off-Diagonal)
relationship_system_prompt = """You are a financial analyst. Your task is to read a news article and find the single, most important semantic relationship between two specified companies.

Your goal is to respond ONLY with a JSON object containing three keys:

1.  **`relationship_type`**: A single category for the relationship.
    * Examples: 'Merger', 'Partnership', 'ExclusiveSupplier', 'Competition', 'Lawsuit', 'Collaboration', 'IndustryTrend', 'MarketNoise', 'NoRelationship'

2.  **`sentiment_score`**: A qualitative score from -1.0 to +1.0 based on the relationship's type and severity.

3.  **`explanation`**: A brief, one-sentence explanation for your score and type, citing the specific event (e.g., "Assigning -0.7 for 'Lawsuit' due to the patent infringement filing.").

**Scoring Guidelines:**
* You must distinguish between financially material events and non-material noise.
* Assign `0.0` or minimal scores (e.g., +/- 0.1) with type 'MarketNoise' for non-financial events, such as a CEO's personal life, political opinions, or generic market reports that mention both companies without a direct interaction.
* Focus only on the score for events that clearly impact revenue, costs, strategy, or competition.

+1.0 (Transformative / Merger): "Apple and Google announce a full merger."
+0.9 (Deep, Exclusive Partnership): "NVIDIA announces it is now the exclusive chip supplier for all of Apple's future AI."
+0.8 (Major R&D Breakthrough): "Johnson & Johnson announces a successful Phase 3 trial for a new blockbuster drug."
+0.7 (Major Contract / Product Success): "NVIDIA's new GPU launch sees record-breaking pre-orders."
+0.6 (Positive Management / Strategy): "Apple hires a renowned AI expert from Google as its new CEO."
+0.5 (Standard, Positive Partnership): "Apple and JPMorgan announce a new co-branded credit card."
+0.4 (Vague Positive / Collaboration): "Microsoft and Apple are 'in talks' about AI safety standards."
+0.3 (Minor Positive / Industry Trend): "Apple and Google both benefit from a positive analyst report on the tech sector."
+0.2 (Slightly Positive): "A market report mentions Google and Apple as stable performers."
+0.1 (Barely Positive): Very weak positive signal.
0.0 (Neutral / Noise): "This is a general market report. Stocks mentioned include Apple, Google, and NVIDIA."
-0.1 (Barely Negative): Very weak negative signal.
-0.2 (Slightly Negative): "Analysts are 'cautious' on the tech sector, which includes Apple and Google."
-0.3 (Minor Negative / Industry Trend): "A new report highlights minor supply chain concerns for Apple and NVIDIA."
-0.4 (Vague Competition): "Google's new product is seen as a 'potential' threat to Apple."
-0.6 (Heightened Competition): "Apple and Google are in a 'price war', cutting margins."
-0.7 (Major Lawsuit / Negative Management): "Apple files a significant patent infringement lawsuit against NVIDIA."
-0.8 (Serious Setback / Product Failure): "Google's new phone is recalled due to a major defect."
-0.9 (Severe Event / R&D Failure): "Johnson & Johnson's drug trial is halted by the FDA for safety concerns."
-1.0 (Catastrophic / Existential): "Regulators are moving to break up Google."

Respond ONLY with a JSON object."""

class RelationshipOutput(types.TypedDict):
    published_date: str
    sentiment_score: float
    relationship_type: str 
    explanation: str


relationship_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=RelationshipOutput
)

# --- 3. Load Your JSON File ---
try:
    with open('cleaned_high_signal_articles.json', 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
except FileNotFoundError:
    print("Error: Input File not found.")
    exit()

print(f"Loaded {len(articles_data)} articles from file.")

# --- 4. Loop Through Articles and Call API ---
all_single_entity_outputs = []
all_relationship_outputs = []
total_output_tokens = 0
total_input_tokens = 0
articles_to_process = articles_data

print(f"--- Processing {len(articles_to_process)} articles... ---")

for article in tqdm(articles_to_process, desc="Processing articles"):
    
    article_text = article.get('text')
    entities = article.get('found_entities', [])
    date = article.get('published', 'UNKNOWN_DATE')[:10]
    
    if not entities or not article_text:
        continue # Skip if no entities or no text

    try:
        # === PASS 1: SINGLE-ENTITY SENTIMENT ===
        # Run this for *every* entity found in the article
        for company in entities:
            user_prompt = f"Article: \"{article_text}\"\nDate: \"{date}\"\nCompany: \"{company}\""
            
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[single_entity_system_prompt, user_prompt],
                config=single_entity_config
            )
            
            total_output_tokens += response.usage_metadata.candidates_token_count
            total_input_tokens += response.usage_metadata.prompt_token_count
            
            result = json.loads(response.text)
            result['uuid'] = article.get('uuid')
            result['processed_company'] = company 
            all_single_entity_outputs.append(result)

        # === PASS 2: RELATIONSHIP SENTIMENT ===
        # *If* there are 2 or more entities, also run the relationship pass
        if len(entities) >= 2:
            unique_entities = sorted(list(set(entities)))
            for pair in combinations(unique_entities, 2):
                company_a, company_b = pair
                user_prompt = f"Article: \"{article_text}\"\nDate: \"{date}\"\nCompany A: \"{company_a}\"\nCompany B: \"{company_b}\""
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[relationship_system_prompt, user_prompt],
                    config=relationship_config
                )
                
                total_output_tokens += response.usage_metadata.candidates_token_count
                total_input_tokens += response.usage_metadata.prompt_token_count
                
                result = json.loads(response.text)
                result['uuid'] = article.get('uuid')
                result['processed_pair'] = [company_a, company_b]
                all_relationship_outputs.append(result)

    except Exception as e:
        print(f"\nError processing article (UUID {article.get('uuid')}): {e}")
        pass

# --- 5. Save Your Results ---
output_filename_single = 'single_entity_results.json'
with open(output_filename_single, 'w', encoding='utf-8') as f:
    json.dump(all_single_entity_outputs, f, indent=4)

output_filename_relation = 'relationship_results.json'
with open(output_filename_relation, 'w', encoding='utf-8') as f:
    json.dump(all_relationship_outputs, f, indent=4)

print(f"\n--- DONE ---")
print(f"Successfully saved {len(all_single_entity_outputs)} single entity results to '{output_filename_single}'.")
print(f"Successfully saved {len(all_relationship_outputs)} relationship results to '{output_filename_relation}'.")
print(f"Total input tokens consumed: {total_input_tokens}")
print(f"Total output tokens consumed: {total_output_tokens}")