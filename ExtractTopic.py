import spacy

# Load the English language model in spaCy
nlp = spacy.load("en_core_web_sm")

# Function to extract the overarching topic from a given phrase
def extract_overarching_topic(phrase):
    doc = nlp(phrase)
    
    # Get the main verb of the sentence to identify the action or topic
    main_verb = [token for token in doc if token.pos_ == "VERB"][0]
    
    return main_verb.lemma_

# Get the input phrase from the user
input_phrase = input("Enter a phrase: ")

# Extract the overarching topic from the input phrase
overarching_topic = extract_overarching_topic(input_phrase)

print("Overarching Topic:", overarching_topic)