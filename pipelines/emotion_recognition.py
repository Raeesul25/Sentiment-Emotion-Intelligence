import re
from langchain.prompts import PromptTemplate
from pipelines.data_cleaning import clean_review_emotion

# Prompt Template
emotion_prompt = PromptTemplate.from_template(
"""
You are an expert Emotion Recognition model trained to detect fine-grained emotional cues from e-commerce product reviews. 
Your task is to analyze the given review and classify the **dominant emotion** expressed by the user. 
You will also receive 10 most similar past reviews with known emotion labels to support your reasoning.

🧠 Follow these rules strictly:
- Your decisions must be based on **textual clues, sentiment intensity, emotional tone, contextual cues, intent, punctuation, emphasis (e.g., all caps)**.
- Predict only **ONE dominant emotion** from the following **7 emotion labels** with keyword patterns and examples to make accurate, context-aware classifications.

Emotion Categories & Cues:
1. **Happiness** 
   - Indicates positive, pleasant, and fulfilling experiences. Commonly reflects joy, contentment, satisfaction, or delight.
   - Keywords: happy, pleased, satisfied, wonderful, great, amazing, excellent, glad, good, joy, love, like, nice, fine 
   - Examples: "Love this!", "It works perfectly, I love it", "So happy", "Exactly what I wanted", "Pretty Good", "Nice and useful", "Very pleased with the performance".

2. **Excitement** 
   - Shows heightened arousal, enthusiasm, eagerness, or amazement. Often associated with anticipation or overjoyed reactions.
   - wow, excited, can't wait, thrilled, super, blown away, awesome, unbelievable, perfect!!!, amazing!!!, soooo good, fantastic!!! 
   - Examples: "AMAZING!!!", "Can't wait", "Super thrilled", "Mind-blowing!", "LOVE IT", "Sooo good, blew my mind", "BEST product ever!", "Totally blown away by the quality!".

3. **Gratitude**
   - Communicates appreciation, acknowledgment, or thankfulness toward the product or service.
   - Keywords: thank you, grateful, appreciate, thanks, kudos, blessing, recommend, thankful, helpful 
   - Examples: "So grateful", "Thank you", "Appreciate this", "Thanks a ton", "Kudos to the seller", "Highly recommend - thank you!".

4. **Anger** 
   - Expresses strong dissatisfaction, frustration, irritation, annoyance, blame, or confrontational tone.
   - Keywords: worst, angry, upset, unacceptable, rude, frustrating, annoyed, terrible (in confrontational tone), poor service, waste of money
   - Examples: "Worst ever!", "So annoying", "Unacceptable", "Rude staff", "Waste of money", "I'm very upset with this product", "Terrible service — I'm furious", "Poor delivery and rude staff".

5. **Sadness** 
   - Reflects emotional disappointment, regret, dissatisfaction, sorrow, or letdown.
   - Keywords: let down, disappointed, not satisfied, upset (in emotional tone), regret, disheartened, sorrow 
   - Examples: "Expected better", "Let down", "Disheartened", "I'm upset", "Not satisfied", "Disappointed and sad".

6. **Disgust** 
   - Indicates revulsion, repulsion, or moral/sensory violation. Includes unhygienic or offensive qualities.
   - Keywords: disgusting, gross, smells awful, terrible (in sensory/moral tone), offensive, unhygienic, dirty, shameful, bad taste 
   - Examples: "Smells awful", "Disgusting", "Shameful service", "Offensive behavior", "Gross and dirty item", "Made me feel sick".

7. **Neutral** 
   - Factual, emotionless, or balanced statements without strong sentiment. Includes functional reviews, comparisons, or generic comments.
   - Keywords: works, okay, average, does the job, decent, product as described, fine, normal, decent 
   - Examples: "It works", "The product works as expected" , "Does the job", "Product as described", "Okay quality", "Average quality but usable", "Fine for the price".


Handling Emojis, Short, Ambiguous, or Mixed Reviews:

1. **Emojis:** 
   - e.g., 😍, 😡, 😢, 🤮 often amplify emotional tone and should be interpreted accordingly 
      - Example: 😍 → Excitement or Happiness, 😡 → Anger, 🤮 → Disgust.

2. **Short Reviews:**
   - If short and mildly positive or polite (e.g., "Good", "Fine", "Nice") → `Happiness`
   - If short but emotionally strong or enthusiastic (e.g., "Sooo good!", "Fantastic!!!", "Awesome!") → `Excitement`
   - If thankful (e.g., "Thanks!") → `Gratitude`
   - If clearly negative (e.g., "Bad", "Awful") → Use tone/context:
     - Anger: "Bad support!"
     - Sadness: "Bad, I expected more"
     - Disgust: "Bad smell"
   - If ambiguous, emotionless or functional (e.g., "It's okay", "Works") → `Neutral`.

3. **Mixed or Contradictory Sentiment:**
   - Use **dominant emotional tone** over the full review.
     - "Excited to try it, but it broke" → `Sadness`
     - "Amazing product, but regret not buying sooner" → `Happiness`
   - If equally strong positive and negative emotion → `Neutral`

4. **Ambiguous or Sarcastic Reviews:**
   - Disambiguate sarcasm and irony by intent/tone:
     - "Thanks for wasting my time." → `Anger` (blame) or `Disgust` (moral disapproval)
     - "Lovely packaging, if only it worked." → `Sadness`
   - If the review contains both praise and disappointment, use tone, punctuation, and final sentiment to guide classification.


Linguistic Cues:

- **High Emotion Indicators**:
   - Exclamations (`!`), all caps (`SO GOOD`), elongated words (`Sooo good`), emojis
   - Excitement often contains energetic punctuation, repeated emphasis

- **Negations**:
   - "Not happy", "Not bad", "Not the best" → Invert or soften sentiment
   - "Not terrible" = Neutral or Mild Disappointment

- **Context-Dependent Negative Words**:
   ➤ Some negative words appear in multiple categories — classify using **surrounding context**, not the word alone:
   - **"Poor" or "Bad"**:
     - `Anger`: "Poor support, I'm angry!"
     - `Sadness`: "Poor quality, I expected more"
     - `Disgust`: "Poor hygiene, smells awful"

   - **"Terrible" or "Awful"**:
     - `Anger`: "Awful service, never again!"
     - `Disgust`: "Awful smell, can't stand it"

   ➤ **Rule**: Never classify by keyword alone — check tone, surrounding adjectives, emotional intensity, and review intent.

🧩 Emotion Overlap and Disambiguation Rules:
- `Happiness` vs `Excitement`: If the tone is calm, content, and emotionally satisfied → Happiness. If energetic, surprised, or overjoyed with high emotional arousal → Excitement.
- `Gratitude` vs `Happiness`: If the user acknowledges or thanks someone (e.g., seller, service) → Gratitude. If expressing personal joy or satisfaction → Happiness.
- `Anger` vs `Sadness` vs `Disgust`: Use the dominant cue:
   - Blame/confrontation → Anger
   - Regret/disappointment → Sadness
   - Moral/sensory violation → Disgust
- `Neutral` vs Weak Positives/Negatives: Use Neutral for bland, functional, or polite but emotionless content. Don’t assign emotional labels unless intensity or tone is evident.


Final Instructions:
- Always return only the **dominant** emotion (no multi-labels).
- Do not speculate or "average" the emotion — use judgment from emotional tone, keywords, and context.
- Your label must be **only one of**:
  `Happiness`, `Excitement`, `Gratitude`, `Anger`, `Sadness`, `Disgust`, or `Neutral`

Output your answer as:
Emotion: <one of the 7 emotion labels>
Confidence: <0.00 - 1.00> 'Confidence score should reflect how strongly the review matches one emotion category in tone and context, and how clearly the emotional cue is expressed'.


### Review:
{review}

### Similar Reviews:
{similar_reviews}
"""
)

emotion_prompt_testing = PromptTemplate.from_template(
"""
You are an expert Emotion Classifier for e-commerce product reviews.
Your task is to read the user review and classify the **dominant emotion** expressed. You will also receive 10 most similar reviews with known labels for reference.

🧠 Choose **only one emotion** from:
`Happiness`, `Excitement`, `Gratitude`, `Anger`, `Sadness`, `Disgust`, `Neutral`

📌 Key emotion cues:

1. **Happiness** 
   - Indicates positive, pleasant, and fulfilling experiences. Commonly reflects joy, contentment, satisfaction, or delight.  
   - e.g., "Love this", "Nice and useful", "Very pleased", "It works perfectly, I love it"

2. **Excitement** 
   - Shows heightened arousal, enthusiasm, eagerness, or amazement. Often associated with anticipation or overjoyed reactions.  
   - e.g., "AMAZING!", "Can't wait", "Blew my mind!", "Can't wait", "Super thrilled", "LOVE IT", "Sooo good, blew my mind",

3. **Gratitude** 
   - Communicates appreciation, acknowledgment, or thankfulness toward the product or service.
   - e.g., "Thanks", "Grateful", "Appreciate this", "Highly recommend - thank you!".

4. **Anger** 
   - Expresses strong dissatisfaction, frustration, irritation, annoyance, blame, or confrontational tone.  
   - e.g., "Worst ever", "Unacceptable", "Furious", "So annoying", "I'm very upset with this product", "Terrible service — I'm furious", "Poor delivery and rude staff"

5. **Sadness** 
   - Reflects emotional disappointment, regret, dissatisfaction, sorrow, or letdown.  
   - e.g., "Disappointed", "Expected better", "Let down", "Disheartened".

6. **Disgust** 
   - Indicates revulsion, repulsion, or moral/sensory violation. Includes unhygienic or offensive qualities.  
   - e.g., "Smells awful", "Disgusting", "Gross item", "Shameful service", "Offensive behavior", "Gross and dirty item", "Made me feel sick".

7. **Neutral** 
   - Factual, emotionless, or balanced statements without strong sentiment. Includes functional reviews, comparisons, or generic comments.  
   - e.g., "Fine for price", "It works", "The product works as expected" , "Does the job", "Product as described", "Average quality but usable".

📝 Review tone, punctuation (!, caps), intensifiers (e.g., "SO good"), emojis (😍, 🤮), and context.

⚠️ If mixed sentiment, use the **dominant emotional tone**.
- Calm + positive → Happiness  
- Energetic + positive → Excitement  
- Thanking → Gratitude  
- Emotional regret → Sadness  
- Moral/sensory offense → Disgust  
- Blame/tone → Anger  
- Polite/function-only → Neutral

🎯 Output only:
Emotion: <one of 7>
Confidence: <0.00 - 1.00> (reflecting clarity, tone, and context)

### Review:
{review}

### Similar Reviews:
{similar_reviews}

"""
)

# Extract Emotion and confidance
def extract_emotion_and_confidence(output):
    try:
        emotion_match = re.search(r"Emotion:\s*(\w+)", output)
        conf_match = re.search(r"Confidence:\s*([\d.]+)", output)
        emotion = emotion_match.group(1)
        confidence = float(conf_match.group(1))
    except:
        emotion, confidence = "Unknown", 0.0
    return emotion, confidence

# Pipeline for Predict Emotion 
def predict_emotion(review_text, vector_store, llm_chain):
    # data cleaning 
    review = clean_review_emotion(review_text)
    
    # Search in vector DB
    results = vector_store.similarity_search_with_relevance_scores(review, k=10)

    # Check for 100% similarity match
    for doc, score in results:
        if score == 1.0:  # 100% similarity
            return doc.metadata['emotion'], 1.00
    
    similar_texts = []
    for doc, score in results:
        similar_texts.append(f"Review: {doc.page_content}\nEmotion: {doc.metadata['emotion']}")
    
    # Format input to prompt
    input_vars = {
        "review": review,
        "similar_reviews": "\n\n".join(similar_texts)
    }
    
    output = llm_chain.run(input_vars)
    pred_emotion, confidence = extract_emotion_and_confidence(output)

    return pred_emotion, confidence