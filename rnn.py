import io
import os
import unicodedata
import string
import glob
import random

import zipfile
import urllib.request

import torch
import torch.nn as nn

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



print("Version ================================================")
print(f"Torch version: {torch.__version__}")
print("\n\n")



print("Device configuration =======================================")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("\n\n")




print("Download and extract dataset ==============================")
url = "https://download.pytorch.org/tutorial/data.zip"

zip_path = "data.zip"
extract_path = os.getcwd()

# download only if data.zip doesn't exist
if not os.path.exists(zip_path):
    print("Downloading data.zip...")
    urllib.request.urlretrieve(url, zip_path)
    print("data.zip downloaded successfully!")
else:
    print("data.zip already exists. Skipping download.")

# extract zip file
print("Extracting data.zip...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)
print("data.zip extracted successfully!")
print("\n\n")




print(f"Hyperparameters ======================================")
# alphabet small + capital letters + " .,;'"
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)
RANDOM_SEED = 42

print(f"Alphabet: {ALL_LETTERS}")
print(f"Number of letters: {N_LETTERS}")
print("\n\n")




print(f"Helper functions =======================================")
# function to turn a unicode string to plain ASCII
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn'
      and c in ALL_LETTERS)

# find letter index from ALL_LETTERS, e.g. "a" = 0
def letter_to_index(letter):
  return ALL_LETTERS.find(letter)

# turn a line into an array of one-hot letter vectors
def line_to_tensor(line):
  tensor = torch.zeros(len(line), 1, N_LETTERS)
  for i, letter in enumerate(line):
      tensor[i][0][letter_to_index(letter)] = 1
  return tensor

print(f"Unicode `Ślusàrski` to ASCII: {unicode_to_ascii('Ślusàrski')}")

print(f"Letter `J` to index: {letter_to_index('J')}")

print(f"Line `Jones` tensor size: {line_to_tensor('Jones').size()}")
print(f"Line `Jones` to tensor: {line_to_tensor('Jones')}")
print("\n\n")




print(f"Loading data ==================================")
data = []

for filename in glob.glob("data/names/*.txt"):
  class_name = os.path.splitext(os.path.basename(filename))[0]

  with io.open(filename, encoding="utf-8") as file:
    names = file.read().strip().split("\n")

  for name in names:
    name = unicode_to_ascii(name)
    data.append([name, class_name])

df = pd.DataFrame(data, columns=["name", "class_name"])
print(f"Dataframe head: {df.head()}")
print("\n\n")




print("Train- test split ====================================")

X= df["name"]
y= df["class_name"]

# split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=RANDOM_SEED)

# reset index
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# print data sizes
print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")
print("\n\n")




# create classes array
print("Create classes array ===============================")
classes = df["class_name"].unique()
print(f"Classes: {classes}")
print("\n\n")





# create custom rnn class #####################################
class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.hidden_size = hidden_size

    self.i2h = nn.Linear(in_features= input_size+ hidden_size,
                         out_features= hidden_size)
    self.i2o = nn.Linear(in_features= input_size+ hidden_size,
                         out_features= output_size)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, input_tensor, hidden_tensor= None):

    if hidden_tensor is None:
      hidden_tensor = torch.zeros(1, self.hidden_size).to(device)

    combined= torch.cat((input_tensor, hidden_tensor), dim=1)

    hidden= self.i2h(combined)
    output= self.i2o(combined)
    output= self.softmax(output)
    return output, hidden




print(f"Create model ===========================================")

N_HIDDEN = 128
N_CLASSES = len(classes)

torch.manual_seed(RANDOM_SEED)

model = RNN(input_size= N_LETTERS,
            hidden_size= N_HIDDEN,
            output_size= N_CLASSES)

model.to(device)

print(f"Model: {model}")
print(f"Model device: {next(model.parameters()).device}")
print("\n\n")





print(f"Create loss function and optimizer ==========================")

torch.manual_seed(RANDOM_SEED)

loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.005 )
print(f"Loss function: {loss_fn}")
print(f"Optimizer: {optimizer}")
print("\n\n")




# accuracy function ##########################################
def accuracy_fn(y_true, y_pred):
  correct = 0
  for i in range(len(y_pred)):
    if y_pred[i].item() == y_true[i].item():
      correct+= 1
  acc = (correct / len(y_pred)) * 100
  return acc




print("Model evaluation =====================================")
def eval_model(model: nn.Module,
               X_test= X_test,
               y_test= y_test,
               loss_fn : nn.NLLLoss= loss_fn,
               device = device):
  current_test_loss, current_test_acc = 0, 0

  # move model to target device
  model.to(device)

  # test step
  for line in range(len(X_test)):
    # evaluation mode
    model.eval()

    with torch.inference_mode():

      # convert to torch.Tensor
      line_tensor = line_to_tensor(X_test[line])
      category_tensor = torch.tensor([classes.tolist().index(y_test[line])],
                                     dtype=torch.long)

      # move to  target device
      line_tensor, category_tensor = line_tensor.to(device), category_tensor.to(device)

      # forward pass
      hidden= None
      for letter in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[letter], hidden)

      # calc loss
      test_loss = loss_fn(output, category_tensor)
      current_test_loss += test_loss.item()

      # calculate accuracy
      test_pred = output.argmax(dim=1)
      current_test_acc += accuracy_fn(y_true=category_tensor,
                                      y_pred=test_pred)

  # calculate avg test loss
  test_loss_value = current_test_loss / len(X_test)

  # calculate average test accuracy
  test_acc_value = current_test_acc / len(X_test)

  print(f'Test loss: {test_loss_value:.4f} | Test accuracy: {test_acc_value:.2f}%')

eval_model(model, X_test, y_test, loss_fn, device)
print("\n\n")





print("Training loop =============================================")
current_train_loss = 0
current_test_loss = 0
train_loss_values = []
test_loss_values = []

current_train_acc = 0
current_test_acc = 0
train_acc_values = []
test_acc_values = []

torch.manual_seed(RANDOM_SEED)

epochs = 5

for epoch in range(epochs):

  # train step
  for line in range(len(X_train)):

    # train mode
    model.train()

    # convert name and culture to tensors
    line_tensor = line_to_tensor(X_train[line])
    category_tensor = torch.tensor([classes.tolist().index(y_train[line])],
                                   dtype=torch.long)

    line_tensor, category_tensor = line_tensor.to(device), category_tensor.to(device)

    # forward pass
    hidden= None
    for letter in range(line_tensor.size()[0]):
      output, hidden = model(line_tensor[letter], hidden)

    # calculate loss
    loss = loss_fn(output, category_tensor)
    current_train_loss += loss.item()

    # calculate acc
    train_pred = output.argmax(dim=1)
    current_train_acc += accuracy_fn(y_true=category_tensor,
                                    y_pred=train_pred)

    # zero grad
    optimizer.zero_grad()

    # backpropagation
    loss.backward()

    # optimizer step
    optimizer.step()

  # calc average train loss
  train_loss_value= current_train_loss / len(X_train)
  train_loss_values.append(train_loss_value)
  current_train_loss = 0

  # calculate average train accuracy
  train_acc_value = current_train_acc / len(X_train)
  train_acc_values.append(train_acc_value)
  current_train_acc = 0
  print(f'Epoch: {epoch+1} | Train loss value: {train_loss_value:.4f} | Train accuracy value: {train_acc_value:.2f}%')

  # test step
  for line in range(len(X_test)):
    # evaluation mode
    model.eval()
    with torch.inference_mode():

      # convert to torch.Tensor
      line_tensor = line_to_tensor(X_test[line])
      category_tensor = torch.tensor([classes.tolist().index(y_test[line])],
                                     dtype=torch.long)

      line_tensor, category_tensor = line_tensor.to(device), category_tensor.to(device)

      # forward pass
      hidden = None
      for letter in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[letter], hidden)

      # calc loss
      test_loss = loss_fn(output, category_tensor)
      current_test_loss += test_loss.item()

      # calculate accuracy
      test_pred = output.argmax(dim=1)
      current_test_acc += accuracy_fn(y_true=category_tensor,
                                      y_pred=test_pred)

  # calculate avg test loss
  test_loss_value = current_test_loss / len(X_test)
  test_loss_values.append(test_loss_value)
  current_test_loss = 0


  # calculate average test accuracy
  test_acc_value = current_test_acc / len(X_test)
  test_acc_values.append(test_acc_value)
  current_test_acc = 0

  print(f'Epoch: {epoch+1} | Test loss value: {test_loss_value:.4f} | Test accuracy value: {test_acc_value:.2f}%')
print("\n\n")






# plot train and test loss #########################################
plt.figure(figsize=(10, 6))

plt.plot(train_loss_values, label="Train Loss", marker="o")
plt.plot(test_loss_values, label="Test Loss", marker="o")

plt.title("Training and Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.show()






# plot train and test accuracy ######################################
plt.figure(figsize=(10, 6))

plt.plot(train_acc_values, label="Train Accuracy", marker="o")
plt.plot(test_acc_values, label="Test Accuracy", marker="o")

plt.title("Training and Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.show()




# make prediction function #############################
def make_prediction(input_line):
  model.eval()
  with torch.inference_mode():
    line_tensor = line_to_tensor(input_line)
    line_tensor = line_tensor.to(device)

    hidden = None
    for i in range(line_tensor.size()[0]):
      output, hidden = model(line_tensor[i], hidden)

    return output




print(f"Test model ===========================================")
name = "Sohaib"
y_logit = make_prediction(name)
y_prob = torch.softmax(y_logit, dim=1)
y_pred = y_prob.argmax(dim=1)
class_name = classes[y_pred.item()]
print(f"Class of the name `{name}` is: {class_name}")
print("\n\n")


