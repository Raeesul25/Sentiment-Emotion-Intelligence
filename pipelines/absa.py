import json
import ast
from collections import defaultdict
from pipelines.data_cleaning import clean_review_absa

# Few-shot example selection per category
def select_diverse_fewshot_examples(category_df, k=10):
    aspect_sentiment_map = defaultdict(list)
    
    # Group reviews by (aspect, sentiment) to ensure diversity
    for _, row in category_df.iterrows():
        try:
            label_dict = json.loads(row["Decisions"])  # Convert string to dict
        except:
            label_dict = row["Decisions"]  # In case it's already a dict

        review = row["Cleaned_Reviews"]
        
        # Store the full label_dict under a unique key (tuple of sorted items)
        key = tuple(sorted(label_dict.items()))  # To uniquely identify this combination
        
        aspect_sentiment_map[key].append(review)
    
    # Shuffle and select up to k diverse examples
    selected = []
    used_keys = set()
    
    for key, reviews in aspect_sentiment_map.items():
        if len(selected) >= k:
            break
        if key not in used_keys:
            # Convert tuple of tuples to a dictionary
            label_dict = dict(key)
            selected.append((reviews[0], label_dict))  # pick one example per group
            used_keys.add(key)
    
    # Fill remaining slots with random samples if needed
    while len(selected) < k:
        row = category_df.sample(1).iloc[0]
        try:
            label_dict = json.loads(row["Decisions"])
        except:
            label_dict = row["Decisions"]
        selected.append((row["Cleaned_Reviews"], label_dict))
    
    return selected[:k]


def build_prompt(review, category, train_df):
    category_examples = train_df[train_df["Category"] == category]
    fewshot_examples = select_diverse_fewshot_examples(category_examples, k=10)

    prompt = (
"""
You are an expert in Aspect-Based Sentiment Analysis (ABSA) for e-commerce reviews.
Extract product-related aspects (explicit or implicit) and assign sentiment ("Positive", "Negative", "Neutral").

⚠️ Only extract meaningful aspects related to the product's qualities or functions. Ignore delivery experience, pricing (unless about value), seller or service feedback, and unrelated opinions.

📌 Output Format (per review):
{
  "Aspect 1": "Sentiment",
  "Aspect 2": "Sentiment",
  ...
}

🎯 Common Aspects Across Categories:
- **General**: Quality, Durability, Design, Size, Color, Fit, Packaging, Instructions, Functionality, Value, Material, Image Accuracy
- **Electronics**: Battery Life, Sound Quality, Display, Performance, Connectivity, Portability, Build, Accessories
- **Clothing**: Comfort, Stretch, Fabric, Fit, Style, Stitching
- **Musical Instruments**: Tuning Stability, Tone, Sustain, Playability, Craftsmanship
- **Toys & Games**: Safety, Educational Value, Engagement, Durability
- **Health & Beauty**: Effectiveness, Scent, Skin Feel, Absorption, Sensitivity
- **Pet Supplies**: Chew Resistance, Safety, Pet Engagement, Allergic Reaction
- **Home & Garden**: Assembly, Cleaning Ease, Sturdiness, Weather Resistance
- **Office Supplies**: Ink Quality, Print Clarity, Ergonomics, Grip, Paper Compatibility

🧠 Special Cases:

1. **No clear aspect** → Use `"Overall Quality": "Sentiment"`
   - "Works well!" → {"Overall Quality": "Positive"}

2. **Sarcasm / Implicit sentiment**
   - "Fantastic. Broke in two days." → {"Durability": "Negative"}
   - "It's supposed to glow, but it doesn't." → {"Functionality": "Negative"}

3. **Mixed Sentiment**
   - "Comfortable shoes, but stitching came loose." → {"Comfort": "Positive", "Stitching": "Negative"}

4. **Image or Expectation Mismatch**
   - "Looks nothing like the ad." → {"Image Accuracy": "Negative"}

5. **Factual with no sentiment** → Assign "Neutral"
   - "Smaller than expected." → {"Size": "Neutral"}

"""
    )

    for example, label_dict in fewshot_examples:
        prompt += f"Review: {example}\n→ ABSA: {json.dumps(label_dict)}\n"

    prompt += f"\nNow, analyze the following review and return the output as a valid JSON object:\nReview: {review}\nABSA:"
    
    return prompt


# Pipeline for Predict ABSA 
def predict_absa(review_text, category, df, llm_chain):
    # data cleaning 
    review = clean_review_absa(review_text)
    
    # Build prompt and get prediction
    prompt = build_prompt(review, category, df)
    response = llm_chain.run(prompt)

    try:
        pred_dict = ast.literal_eval(response.strip())
    except Exception as e:
        pred_dict = {}
    
    return pred_dict
