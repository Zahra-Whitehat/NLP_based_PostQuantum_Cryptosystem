from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer for sequence classification
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Define the input phrase
input_phrase = "Your input phrase here"

# Tokenize the input phrase
inputs = tokenizer(input_phrase, return_tensors="pt")

# Perform inference with the pre-trained BERT model
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()
probabilities = torch.softmax(outputs.logits, dim=1).tolist()[0]

# Output the numerical label and probability scores for each class
print("Predicted Label:", predicted_class)
print("Probability Scores:", probabilities)