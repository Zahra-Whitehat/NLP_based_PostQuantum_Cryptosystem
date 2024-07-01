import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import spacy
from tqdm import tqdm
import torchtext
torchtext.disable_torchtext_deprecation_warning()

# Load spaCy tokenizer
spacy_en = spacy.load('en_core_web_sm')

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Download and load the IMDb dataset
train_iter, test_iter = IMDB(split=('train', 'test'))

# Build vocabulary
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

# Create custom dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, data_iter, vocab, tokenizer):
        self.data = list(data_iter)
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]
        label = 1 if label == 'pos' else 0
        text_pipeline = lambda x: [self.vocab[token] for token in self.tokenizer(x)]
        text = text_pipeline(text)
        return torch.tensor(label, dtype=torch.int64), torch.tensor(text, dtype=torch.int64)

# Create DataLoader
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_label, _text) in batch:
        label_list.append(_label)
        text_list.append(_text)
        lengths.append(len(_text))
    labels = torch.tensor(label_list, dtype=torch.int64)
    texts = pad_sequence(text_list, batch_first=True, padding_value=vocab['<pad>'])
    lengths = torch.tensor(lengths, dtype=torch.int64)
    return labels, texts, lengths

batch_size = 64
train_dataset = TextClassificationDataset(train_iter, vocab, tokenizer)
test_dataset = TextClassificationDataset(test_iter, vocab, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)

# Hyperparameters
INPUT_DIM = len(vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128  # Reduced from 256
OUTPUT_DIM = 1
N_LAYERS = 1  # Reduced from 2
BIDIRECTIONAL = True
DROPOUT = 0.3  # Reduced from 0.5
PAD_IDX = vocab['<pad>']

# Initialize the model
model = LSTMClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

# Load pre-trained embeddings
glove_vectors = GloVe(name='6B', dim=EMBEDDING_DIM)
model.embedding.weight.data.copy_(glove_vectors.get_vecs_by_tokens(vocab.get_itos()))
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# Training setup
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

# Training loop
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

N_EPOCHS = 3  # Reduced from 5

for epoch in range(N_EPOCHS):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for labels, texts, lengths in tqdm(train_dataloader, desc=f'Epoch {epoch+1}'):
        optimizer.zero_grad()
        texts, lengths = texts.to(device), lengths.to(device)
        labels = labels.float().to(device)
        predictions = model(texts, lengths).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_dataloader)}, Accuracy: {epoch_acc/len(train_dataloader)}')

# Evaluate the model
model.eval()
epoch_loss = 0
epoch_acc = 0

with torch.no_grad():
    for labels, texts, lengths in tqdm(test_dataloader, desc='Evaluating'):
        texts, lengths = texts.to(device), lengths.to(device)
        labels = labels.float().to(device)
        predictions = model(texts, lengths).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        epoch_loss += loss.item()
        epoch_acc += acc.item()

print(f'Test Loss: {epoch_loss/len(test_dataloader)}, Test Accuracy: {epoch_acc/len(test_dataloader)}')

# Function to classify new sentences
def classify_sentence(sentence):
    model.eval()
    tokenized = [tok.text for tok in spacy_en.tokenizer(sentence)]
    indexed = [vocab[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    length_tensor = torch.LongTensor(length).to(device)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

# Example usage
new_sentence = "The movie was fantastic and I really enjoyed it."
print(f'The predicted class for the sentence is: {classify_sentence(new_sentence)}')

# Save the model and optimizer state
torch.save({
    'epoch': N_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    # Add any other information you want to save
}, 'trained_model.pth')

# Load the model and optimizer state
checkpoint = torch.load('trained_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# Optionally load other information from the checkpoint