# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# %matplotlib inline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

df = pd.read_csv("/20191002-items.csv")
df.head()

df.shape

df.info()

sns.countplot(df.averageRating)
plt.xlabel('review score');

def to_sentiment(rating):
    try:
        rating = int(rating)
        if rating <= 2:
            return 0
        elif rating == 3:
            return 1
        else:
            return 2
    except ValueError:
        return None  # or another default value

df['sentiment'] = df['averageRating'].apply(to_sentiment)

class_names = ['negative', 'neutral', 'positive']

ax = sns.countplot(df.sentiment)
plt.xlabel('review sentiment')
ax.set_xticklabels(class_names);

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

sample_txt = "This is an example sentence for tokenization."
encoding = tokenizer.encode_plus(
  sample_txt,
  max_length=32,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,  #RETURNS 0 FOR PADDINGS
  return_tensors='pt',  # Return PyTorch tensors
)

encoding.keys()

print(len(encoding['input_ids'][0]))
encoding['input_ids'][0]

print(len(encoding['attention_mask'][0]))
encoding['attention_mask']

tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

token_lens = []

for txt in df.brandName:
    if isinstance(txt, str):  # Process only valid strings
        tokens = tokenizer.encode(txt, max_length=512, truncation=True)
        token_lens.append(len(tokens))
    else:
        token_lens.append(0)  # Assign length 0 for missing values

print(token_lens)

sns.distplot(token_lens)
plt.xlim([0, 256]);
plt.xlabel('Token count');

MAX_LEN = 160

class GPReviewDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42  # Set a fixed seed for reproducibility

df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

df_train.shape, df_val.shape, df_test.shape

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GPReviewDataset(
    reviews=df.averageRating.to_numpy(),
    targets=df.totalReviews.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

BATCH_SIZE = 16

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

LABEL_MAPPING = {"Positive": 2, "Neutral": 1, "Negative": 0}  # Adjust based on your dataset

class SentimentDataset(Dataset):
    def __getitem__(self, index):
        review = self.reviews[index]
        target = self.targets[index]

        # Convert string labels to integers
        if isinstance(target, str):
            target = LABEL_MAPPING.get(target, 0)  # Default to 0 if not found

        return {
            'review_text': review,
            'targets': torch.tensor(target, dtype=torch.long)  # Now it's an integer
        }

class SentimentDataset(Dataset):
    def __getitem__(self, index):
        review = self.reviews[index]
        target = self.targets[index]

        # Convert target to an integer if it's a string
        if isinstance(target, str):
            try:
                target = int(target)
            except ValueError:
                print(f"Invalid target at index {index}: {target}")  # Debugging
                target = 0  # Assign a default label

        return {
            'review_text': review,
            'targets': torch.tensor(target, dtype=torch.long)
        }

data = next(iter(train_data_loader))
print(data.keys())

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

last_hidden_state, pooled_output = bert_model(
  input_ids=encoding['input_ids'],
  attention_mask=encoding['attention_mask']
)

print(type(last_hidden_state))

from transformers import AutoModel, AutoTokenizer

# Define model name (change it based on your use case)
model_name = "bert-base-uncased"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print("Model loaded successfully!")

sample_text = "Hello, how are you?"

# Tokenize input
inputs = tokenizer(sample_text, return_tensors="pt")

# Forward pass through model
outputs = model(**inputs)

# Extract last hidden state
last_hidden_state = outputs.last_hidden_state

# Print shape
print(last_hidden_state.shape)

bert_model.config.hidden_size

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

outputs = model(**inputs)

# Directly get pooled output (logits for classification)
pooled_output = outputs.logits  # Shape: [batch_size, num_labels]

print(pooled_output.shape)

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

model = SentimentClassifier(len(class_names))
model = model.to(device)

input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
print(type(input_ids), type(attention_mask))
print(type(model))

outputs = model(input_ids, attention_mask)  # Get model output
logits = outputs.logits  # Extract logits from SequenceClassifierOutput
probs = F.softmax(logits, dim=1)  # Apply softmax on logits

print(probs)  # Now it should work

EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  model = model.train()

  losses = []
  correct_predictions = 0

  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

from collections import defaultdict
import time
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Define model and tokenizer
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust num_labels if needed
tokenizer = BertTokenizer.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define number of epochs
EPOCHS = 10  # Change as needed

# Define training function (replace with actual logic)
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, dataset_size):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in data_loader:
        # Implement training logic here
        pass

    return correct_predictions / dataset_size, total_loss / dataset_size

# Define evaluation function
def eval_model(model, data_loader, loss_fn, device, dataset_size):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    for batch in data_loader:
        # Implement evaluation logic here
        pass

    return correct_predictions / dataset_size, total_loss / dataset_size

# Training loop
start_time = time.time()
history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

end_time = time.time()
print(f"Total Execution Time: {end_time - start_time:.2f} seconds")

plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);

test_acc, _ = eval_model(
  model,
  test_data_loader,
  loss_fn,
  device,
  len(df_test)
)

print(test_acc)  # No need for .item()

def get_predictions(model, data_loader):
  model = model.eval()

  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values

def get_predictions(model, data_loader):
    model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits  # ✅ Extract logits first

            _, preds = torch.max(logits, dim=1)  # ✅ Use logits instead of outputs

            probs = F.softmax(logits, dim=1)  # ✅ Use logits for softmax

            review_texts.extend(d["review_text"])
            predictions.extend(preds.cpu().numpy())
            prediction_probs.extend(probs.cpu().numpy())
            real_values.extend(d["targets"].cpu().numpy())

    return review_texts, predictions, prediction_probs, real_values

print(set(y_test))  # See the unique labels in y_test
print(len(set(y_test)))  # Check how many unique labels exist
print(len(class_names))  # Check if class_names matches the number of unique labels

print(classification_report(y_test, y_pred, labels=list(range(71)), target_names=class_names))

print(f"Unique labels in y_test_mapped: {set(y_test_mapped)}")
print(f"Number of classes in y_test_mapped: {len(set(y_test_mapped))}")
print(f"Number of labels in class_names: {len(class_names)}")

class_names = class_names[:len(set(y_test_mapped))]  # Trim class_names

class_names = [f"Class {i}" for i in range(len(set(y_test_mapped)))]

print(classification_report(
    y_test_mapped,
    y_pred_mapped,
    labels=list(set(y_test_mapped)),  # Explicitly specify expected labels
    target_names=class_names
))

def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment');

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

from sklearn.metrics import classification_report

# Check if y_test_mapped and class_names have matching lengths
print(f"Unique labels in y_test_mapped: {set(y_test_mapped)}")
print(f"Number of classes in y_test_mapped: {len(set(y_test_mapped))}")
print(f"Number of labels in class_names: {len(class_names)}")

# Ensure y_test_mapped is mapped to correct class indices
if len(set(y_test_mapped)) != len(class_names):
    print("Warning: Mismatch between y_test_mapped classes and class_names!")

# Define a valid index to check predictions
idx = 0  # Adjust this as needed

# Ensure y_pred_probs is not empty and has valid dimensions
if len(y_pred_probs) > 0 and idx < len(y_pred_probs):
    print(f"Length of y_pred_probs[{idx}]: {len(y_pred_probs[idx])}")
else:
    print("Error: y_pred_probs is empty or idx is out of range!")

# Generate the classification report
try:
    print(classification_report(y_test_mapped, y_pred_mapped, target_names=class_names))
except ValueError as e:
    print(f"Error in classification_report: {e}")
    print(f"Ensure number of unique labels in y_test_mapped matches class_names.")

class_names = class_names[:len(y_pred_probs[idx])]

print(f"Shape of y_pred_probs: {len(y_pred_probs)}")
print(f"Example y_pred_probs[idx]: {y_pred_probs[idx]}")

pred_df = pd.DataFrame({
    'class_names': class_names,
    'values': y_pred_probs[idx]
})

sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
plt.ylabel('sentiment')
plt.xlabel('probability')
plt.xlim([0, 1]);

"""On raw Text"""

review_text = "I love completing my todos! Best app ever!!!"

encoded_review = tokenizer.encode_plus(
  review_text,
  max_length=MAX_LEN,
  add_special_tokens=True,
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',
)

input_ids = encoded_review['input_ids'].to(device)
attention_mask = encoded_review['attention_mask'].to(device)

output = model(input_ids, attention_mask)
logits = output.logits  # Extract logits

_, prediction = torch.max(logits, dim=1)

print(f'Review text: {review_text}')
print(f'Sentiment  : {class_names[prediction.item()]}')
