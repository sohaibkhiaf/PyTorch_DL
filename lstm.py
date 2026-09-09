import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import nltk
from nltk import sentiment
from nltk.corpus import stopwords
from collections import Counter
import re

import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split

# download stopwords using :
# nltk.download('stopwords')


print("Version ==============================================")
print(f"Torch version: {torch.__version__}")
print("\n\n")

RANDOM_SEED= 42




print("Device configuration ==================================")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("\n\n")






print("Loading data ==================================")
df = pd.read_csv("imdb_dataset.csv")

print(f"Data shape: {df.shape}")
print(f"Number of positive samples: {df[df['sentiment'] == 'positive'].shape[0]}")
print(f"Number of negative samples: {df[df['sentiment'] == 'negative'].shape[0]}")
print(f"Data samples: {df.head()}")
print("\n\n")






print("Train- test split ===========================================")
X, y = df['review'].values, df['sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    random_state=RANDOM_SEED)
print(f'Shape of train data: {X_train.shape}')
print(f'Shape of test data: {X_test.shape}')
print("\n\n")






# plot positive vs negative value counts #################################
dd = pd.Series(y_train).value_counts()
sns.barplot(x=np.array(['negative','positive']),
            y=dd.values,
            hue= np.array(['negative','positive']),
            palette=["red", "green"])
plt.show()






# helper functions ############################################
def preprocess_string(s):
    # remove !  ?  .  ,  ;  :  '  "
    s = re.sub(r"[^\w\s]", '', s)
    # remove digits 1 4 5 6 0
    s = re.sub(r"\d", '', s)

    # e.g. s= "Amazing!!123" => s= "Amazing"
    return s

def tokenize(X_train, y_train, X_test, y_test):

  # create vocabulary
  word_list = []

  # remove stop words (the, is , a, an, of, to, in, and, ...)
  # + preprocess words

  stop_words = set(stopwords.words('english'))
  for text in X_train:
    for word in text.lower().split():
      word = preprocess_string(word)
      if word not in stop_words and word != '':
        word_list.append(word)

  # calculate each word's count in a python dict {word: count, ...}
  word_count = Counter(word_list)

  # sorting on the basis of most common words
  vocabulary = sorted(word_count, key=word_count.get, reverse=True)[:120] # max vocabulary size

  # creating a dict
  word_to_id  = {w:i+1 for i,w in enumerate(vocabulary)}

  # tokenize
  train_sequences , test_sequences = [],[]

  # encode train and test sequences
  for text in X_train:
    train_sequences.append([word_to_id[preprocess_string(word)] for word in text.lower().split()
              if preprocess_string(word) in word_to_id.keys()])
  for text in X_test:
    test_sequences.append([word_to_id[preprocess_string(word)] for word in text.lower().split()
              if preprocess_string(word) in word_to_id.keys()])

  # encode train and test labels
  train_labels_encoded = [1 if label =='positive' else 0 for label in y_train]
  test_labels_encoded = [1 if label =='positive' else 0 for label in y_test]

  return np.array(train_sequences, dtype=object), np.array(train_labels_encoded), np.array(test_sequences, dtype=object), np.array(test_labels_encoded), word_to_id






print("Tokenization =====================================")
X_train_encoded, y_train_encoded, X_test_encoded, y_test_encoded, vocabulary = tokenize(X_train,
                                                        y_train,
                                                        X_test,
                                                        y_test)
print(f"X_train encoded samples: \n{X_train_encoded[:3]}")
print(f"y_train encoded samples: \n{y_train_encoded[:3]}")
print(f"X_test encoded samples: \n{X_test_encoded[:3]}")
print(f"y_test encoded samples: \n{y_test_encoded[:3]}")

print(f'Length of vocabulary: {len(vocabulary)}')
print("\n\n")







print("Plot review length vs frequency =============================")
review_len = [len(review) for review in X_train_encoded]

plt.figure(figsize=(10, 6))

plt.hist(review_len, bins=30)

plt.xlabel("Review length (number of tokens)")
plt.ylabel("Frequency")
plt.title("Distribution of Review Lengths")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

pd.Series(review_len).describe()

print("\n\n")

# max sequence length based on plotted review lengths
MAX_SEQ_LEN = 12







# add padding function #################################
def add_padding(sentences, seq_len):

  # initialize padded sequences with zeros
  padded = np.zeros(
      (len(sentences), seq_len),
      dtype=int
  )

  # store the original length of each sequence
  lengths = []

  for i, review in enumerate(sentences):
    # truncate if review is longer than seq_len
    review_array = np.array(review)

    if  len(review_array) == 0:
      # if the review is empty, assign a minimum length of 1
      lengths.append(1)
    else:
      truncated_review = review_array[:seq_len]
      lengths.append(len(truncated_review))
      # right padding
      padded[i, :len(truncated_review)] = truncated_review

  return padded, np.array(lengths)







print("Add right padding =====================================")
# we have very less number of reviews with length > MAX_SEQ_LEN
# so we will consider only those below it
X_train_pad, train_lengths = add_padding(X_train_encoded, MAX_SEQ_LEN)
X_test_pad , test_lengths = add_padding(X_test_encoded, MAX_SEQ_LEN)

print(f"Shape of train data: {X_train_pad.shape}")
print(f"Shape of test data: {X_test_pad.shape}")
print(f"Shape of train labels: {y_train_encoded.shape}")
print(f"Shape of test labels: {y_test_encoded.shape}")

print(f"Train data: {X_train_pad}")
print(f"Test data: {X_test_pad}")

print(f"Train data lengths shape: {train_lengths.shape}")
print(f"Test data lengths shape: {test_lengths.shape}")

print(f"Train data lengths samples: {train_lengths[:5]}")
print(f"Test data lengths samples: {test_lengths[:5]}")
print("\n\n")







print("Create datasets and data loaders ====================================== ")
train_data = TensorDataset(torch.from_numpy(X_train_pad),
                           torch.from_numpy(train_lengths),
                           torch.from_numpy(y_train_encoded))
test_data = TensorDataset(torch.from_numpy(X_test_pad),
                          torch.from_numpy(test_lengths),
                          torch.from_numpy(y_test_encoded))

# batch size
BATCH_SIZE = 50


# make sure to SHUFFLE your data
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)

print(f'Length of train dataloader: {len(train_dataloader)}')
print(f'Length of test dataloader: {len(test_dataloader)}')
print("\n\n")







print("Batch sample ==========================================")
# obtain one batch of training data
X_batch, lengths_batch, y_batch = next(iter(train_dataloader))

print(f'Shape of X_batch: {X_batch.shape}')
print(f'Shape of y_batch: {y_batch.shape}')
print(f'Shape of lengths_batch: {lengths_batch.shape}')

print(f'X_batch: \n{X_batch}')
print(f'y_batch: \n{y_batch}')
print(f'lengths_batch: \n{lengths_batch}')
print("\n\n")








# sentiment lstm model subclass ###########################################
class SentimentLSTM(nn.Module):
  def __init__(
      self,
      vocabulary_size,
      embedding_dim= 128,
      hidden_dim= 128,
      num_layers= 2,
      dropout=0.3 ):

    super().__init__()

    # embedding layer
    self.embedding = nn.Embedding (
        # 1001 (1000 word+ padding )
        num_embeddings= vocabulary_size,
        # idx => embedding result size, e.g. 96 becomes [0.34, -0.26, ..., 0.13] (128 values)
        embedding_dim= embedding_dim,
        # padding representation is 0
        padding_idx=0
    )

    #lstm layers
    self.lstm = nn.LSTM(
        input_size=embedding_dim, #
        hidden_size= hidden_dim, # hidden state size
        num_layers= num_layers,
        batch_first=True,
        bidirectional=True, # num_directions
        dropout=dropout if num_layers >1 else 0,
    )

    # dropout
    self.dropout = nn.Dropout(dropout)

    # fully connected
    self.fc = nn.Linear(hidden_dim*2, 1) # *2 for BiLSTM (num_directions=2)

  def forward(self, x, lengths):

    # [batch_size=50, seqence_size=500 ]
    embedded = self.embedding(x)

    # [batch_size=50, seqence_size=500, embedding_dim=128]
    packed = pack_padded_sequence(
        embedded, # embedded tockens
        lengths.cpu(), # pack_padded_sequence work on cpu
        batch_first=True, # batch_size is left
        enforce_sorted=False # data not sorted from shorter to longer length
    )

    packed_output, (hidden, cell) = self.lstm(packed)

    # print(f"Hidden shape: {hidden.shape}")

    # hidden:
    # [num_layers=2 × num_directions=2 , batch_size=50, hidden_dim=128] = [4, 50, 128]

    # hidden[0] → Layer 1 Forward
    # hidden[1] → Layer 1 Backward
    # hidden[2] → Layer 2 Forward
    # hidden[3] → Layer 2 Backward

    # we need last lstm layer forward+ backward
    forward_hidden = hidden[-2] # [50, 128]
    backward_hidden = hidden[-1] # [50, 128]

    # [batch_size=50, hidden_dim=128 * 2] = [50, 256]
    hidden = torch.cat(
        (forward_hidden, backward_hidden),
        dim=1   # [dim=0: 50, dim=1: 128+ 128]
    )

    hidden = self.dropout(hidden)

    # [batch_size, 1]
    logits = self.fc(hidden)

    return logits.squeeze(1)






# function to calculate accuracy ######################################
def accuracy_fn(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()





# model evaluation function ################################3
def eval_model(model: nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               loss_fn : nn.BCEWithLogitsLoss,
               device = device):

  # move model to device
  model.to(device)

  #
  test_acc_values = []
  test_loss_values = []

  # test step
  test_loss = []
  test_acc = 0

  # eval mode
  model.eval()
  with torch.inference_mode():
    for X_batch, lengths_batch, y_batch in test_dataloader:

      # move to target device
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)
      lengths_batch = lengths_batch.to(device)

      # forward pass
      y_logit = model(X_batch, lengths_batch)

      # calculate loss
      loss_test = loss_fn(y_logit, y_batch.type(torch.float))
      test_loss.append(loss_test.item())

      # calculate accuracy
      acc_test = accuracy_fn(torch.sigmoid(y_logit), y_batch)
      test_acc += acc_test

  epoch_test_loss = np.mean(test_loss)
  epoch_test_acc = test_acc/len(test_dataloader.dataset)

  test_loss_values.append(epoch_test_loss)
  test_acc_values.append(epoch_test_acc)

  print(f'Test loss: {epoch_test_loss:.4f}')
  print(f'Test acc: {epoch_test_acc*100:.2f}')
  print('='*70)







# train step function ############################################
def train_step(model: nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               loss_fn : nn.BCEWithLogitsLoss,
               optimizer: torch.optim.Optimizer,
               device = device,
               clip: int = None):
  # train step
  train_loss = []
  train_acc = 0

  # train mode
  model.train()

  for X_batch, lengths_batch , y_batch in train_dataloader:
    # move to target device
    X_batch, y_batch = X_batch.to(device),  y_batch.to(device)
    lengths_batch = lengths_batch.to(device)

    y_logit = model(X_batch, lengths_batch)

    # calculate the loss
    loss = loss_fn(y_logit, y_batch.type(torch.float))
    train_loss.append(loss.item())

    # calculate the accuracy
    acc = accuracy_fn(torch.sigmoid(y_logit), y_batch)
    train_acc += acc
    print(f"Train step: acc= {acc/ len(y_batch)*100:.2f}% | loss= {loss.item():.4f}")

    # zero grad
    optimizer.zero_grad()

    # perform backprop
    loss.backward()

    # prevent the exploding gradient problem in rnn/ lstm models
    if clip:
      nn.utils.clip_grad_norm_(model.parameters(), clip)

    # optimizer step
    optimizer.step()

  return  np.mean(train_loss), train_acc/len(train_dataloader.dataset)






# test step function ############################################
def test_step(model: nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               loss_fn : nn.BCEWithLogitsLoss,
               device = device):
  # test step
  test_loss = []
  test_acc = 0

  # eval mode
  model.eval()
  with torch.inference_mode():
    for X_batch, lengths_batch, y_batch in test_dataloader:

      # move to target device
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)
      lengths_batch = lengths_batch.to(device)

      # forward pass
      y_logit = model(X_batch, lengths_batch)

      # calculate loss
      loss_test = loss_fn(y_logit, y_batch.type(torch.float))
      test_loss.append(loss_test.item())

      # calculate accuracy
      acc_test = accuracy_fn(torch.sigmoid(y_logit), y_batch)
      test_acc += acc_test
      print(f"Test step: acc= {acc_test/ len(y_batch)*100:.2f}% | loss= {loss_test.item():.4f}")

  return np.mean(test_loss), test_acc/len(test_dataloader.dataset)







# train loop function #######################################
def train_model(model: nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                loss_fn : nn.BCEWithLogitsLoss,
                optimizer: torch.optim.Optimizer,
                device= device,
                epochs: int = 5,
                clip: int = 5):

  train_loss_values, test_loss_values = [],[]
  train_acc_values, test_acc_values = [],[]

  for epoch in range(epochs):

    # train step
    epoch_train_loss, epoch_train_acc = train_step(model= model,
                                      train_dataloader =train_dataloader,
                                      loss_fn= loss_fn,
                                      optimizer= optimizer,
                                      device= device,
                                      clip = clip)
    # test step
    epoch_test_loss, epoch_test_acc = test_step(model=model,
                                      test_dataloader= test_dataloader,
                                      loss_fn= loss_fn,
                                      device= device)

    # append values for plotting
    train_loss_values.append(epoch_train_loss)
    train_acc_values.append(epoch_train_acc)

    test_loss_values.append(epoch_test_loss)
    test_acc_values.append(epoch_test_acc)

    # print epoch loss and acc
    print(f'Epoch: {epoch+1}')
    print(f'Train loss= {epoch_train_loss:.4f} | Test loss= {epoch_test_loss:.4f}')
    print(f'Train accuracy= {epoch_train_acc*100:.2f}% | Test accuracy= {epoch_test_acc*100:.2f}%')
    print('='*70)

  return train_loss_values, test_loss_values, train_acc_values, test_acc_values






# plot loss and accuracy function ###################################
def plot_loss_and_acc(train_loss_values,
                      test_loss_values,
                      train_acc_values,
                      test_acc_values,
                      epochs):
  fig = plt.figure(figsize = (20, 6))
  # loss
  plt.subplot(1, 2, 2)
  plt.plot(train_loss_values, label='Train loss')
  plt.plot(test_loss_values, label='Test loss')
  plt.title("Loss")
  plt.xticks(range(0, epochs+1, 1))
  plt.legend()
  plt.grid()
  # accuracy
  plt.subplot(1, 2, 1)
  plt.plot(train_acc_values, label='Train Accuracy')
  plt.plot(test_acc_values, label='Test Accuracy')
  plt.title("Accuracy")
  plt.xticks(range(0, epochs+1, 1))
  plt.legend()
  plt.grid()

  plt.show()






print("Train model =======================================")
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

model = SentimentLSTM(vocabulary_size=len(vocabulary) +1, # +1 for padding =0
                      embedding_dim=128,
                      hidden_dim=128 ,
                      num_layers=2,
                      dropout=0.3)

#moving to gpu
model.to(device)

print(f"Model: \n{model}")

# loss and optimization functions
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001)

print(f"Loss function: \n{loss_fn}")
print(f"Optimizer: \n{optimizer}")

print(f"Pre-train evaluation: ")
eval_model(model, test_dataloader, loss_fn, device)

# train
print(f"Training loop: ")
train_loss_values, test_loss_values, train_acc_values, test_acc_values = train_model(model,
                                                  train_dataloader,
                                                  test_dataloader,
                                                  loss_fn,
                                                  optimizer,
                                                  device,
                                                  epochs= 5,
                                                  clip= 5
)
print("\n\n")







# plot loss and acc ######################################
plot_loss_and_acc(train_loss_values,
                  test_loss_values,
                  train_acc_values,
                  test_acc_values,
                  epochs= 5)

print("Post- train evaluation: ")
eval_model(model, test_dataloader, loss_fn, device)






# predict sentiment function #############################
def predict_text(text: str, model: torch.nn.Module):
  # eval mode
  model.eval()
  with torch.inference_mode():

    # preprocess sequence
    word_seq = np.array([vocabulary[preprocess_string(word)] for word in text.split()
                    if preprocess_string(word) in vocabulary.keys()])
    word_seq = np.expand_dims(word_seq, axis=0)

    # add padding
    padded, lengths = add_padding(word_seq, MAX_SEQ_LEN)

    # convert to tensor
    padded, lengths =  torch.from_numpy(padded), torch.from_numpy(lengths)

    # move to device
    padded = padded.to(device)

    # calculate probability
    y_logit = model(padded, lengths)
    y_pred= torch.sigmoid(y_logit)
  return y_pred.cpu().detach().numpy()








print("Test model ========================================")
while True:
  print("Write a review: ")
  text = input()
  if text == 'exit':
    break

  # make prediction
  y_prob = predict_text(text, model)

  # translate to status
  status = "positive" if y_prob > 0.5 else "negative"

  # percentage negative /positive
  prob = (1 - y_prob.item()) if status == "negative" else y_prob.item()

  print(f'Predicted sentiment is {status} with a probability of {prob*100:.2f}%')

print("\n\n")







print("Test model on different dataset =====================================")
df_old = pd.read_csv("imdb_dataset_2.csv")
X, y = df_old['review'].values, df_old['sentiment'].values


correct = 0
total = len(y)
for i in range(len(y)):

    review = X[i]
    sentiment = y[i]
    sentiment_code = 1 if sentiment == "positive" else 0

    y_prob = predict_text(review, model)
    y_pred = torch.round(torch.from_numpy(y_prob))

    if y_pred == sentiment_code:
        correct += 1
        print(f"Correct prediction => {correct} / {total}")
    else:
        print(f"Incorrect prediction")

print(f"Accuracy: {correct/total*100:.2f}%")
print("\n\n")
