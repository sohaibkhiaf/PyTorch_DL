import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



print("Version ==========================================")
print(f"PyTorch version: {torch.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print("\n\n")



print("Device ==========================================")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("\n\n")



print("Create circles ====================================")
n_samples = 1000
X, y= make_circles(n_samples,
                   noise=0.03,
                   random_state=42)

print(f"Length of X= {len(X)}")
print(f"Length of y= {len(y)}")
print(f"Shape of X= {X.shape}")
print(f"Shape of y= {y.shape}")

# print first 5 circles
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
print("First 5 rows in circles dataset: ")
print(circles.head(5))

# print how many circles are 1 / 0
print(f"Label value counts: {circles.label.value_counts()}")
print("\n\n")



# plot circles ####################################
plt.scatter(
    x= X[:, 0],
    y= X[:, 1],
    c=y, # color based on labels
    cmap=plt.cm.RdYlBu # red yellow blue
)
plt.show()


print("Convert data to float32 ==============================")
X= torch.from_numpy(X).type(torch.float)
y= torch.from_numpy(y).type(torch.float)
print(f"Datatype of X= {X.dtype}")
print(f"Datatype of y= {y.dtype}")
print("\n\n")



print("Train- test split ====================================")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Length of X_train= {len(X_train)}")
print(f"Length of X_test= {len(X_test)}")
print(f"Length of y_train= {len(y_train)}")
print(f"Length of y_test= {len(y_test)}")
print("\n\n")


# create model subclass #####################################
class CircleModel(nn.Module):
  def __init__(self):
    super().__init__()
    # in_features X1, X2
    self.layer_1 = nn.Linear(in_features=2, out_features=10)

    # hidden layer
    self.layer_2 = nn.Linear(in_features=10, out_features=10)

    # 0 or 1
    self.layer_3 = nn.Linear(in_features=10, out_features=1)

    # activation (non-linearity )
    self.relu = nn.ReLU()

  def forward(self, x):
    x= self.layer_1(x)
    x= self.relu(x)
    x= self.layer_2(x)
    x= self.relu(x)
    x= self.layer_3(x)
    return x


print("Create model ==========================================")
model = CircleModel().to(device)
print(f"Model state dict: {model.state_dict()}")
print(f"Model device: {next(model.parameters()).device}")
print("\n\n")



# loss function and optimizer ####################################
loss_fn = nn.BCEWithLogitsLoss() # Sigmoid activation built-in
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)



# create accuracy function ######################################
def accuracy_fn(y_true, y_pred):
  correct = 0
  for i in range(len(y_pred)):
    if y_pred[i].item() == y_true[i].item():
      correct+= 1
  acc = (correct / len(y_pred)) * 100
  return acc



print("Evaluate model before training ==============================")
model.eval()
with torch.inference_mode():
  y_logit = model(X_test.to(device))
y_prob = torch.sigmoid(y_logit)
y_pred = torch.round(y_prob)

print(f"Logits: {y_logit[:5].squeeze()}")
print(f"Probs: {y_prob[:5].squeeze()}")
print(f"Pred: {y_pred[:5].squeeze()}")
print(f"Actual: {y_test[:5].squeeze()}")
print(f"Accuracy(5 samples): {accuracy_fn(y_true=y_test[:5], y_pred=y_pred[:5]):.2f}%")
print("\n\n")


print("Training loop ======================================")
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 2000

# move data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

train_loss_values= []
test_loss_values= []
train_acc_values= []
test_acc_values= []
epoch_count= []

for epoch in range(epochs):

  epoch_count.append(epoch)

  # train mode
  model.train()

  # forward pass
  y_logit = model(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logit)) # max logit== max prob

  # calculate loss/ accuracy
  loss = loss_fn(y_logit, y_train)
  acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
  train_loss_values.append(loss.item())
  train_acc_values.append(acc)

  # optimizer zero grad
  optimizer.zero_grad()

  # backpropagation
  loss.backward()

  # gradient descent
  optimizer.step()

  # testing
  model.eval()
  with torch.inference_mode():
    test_logit = model(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logit))

    test_loss = loss_fn(test_logit, y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    test_loss_values.append(test_loss.item())
    test_acc_values.append(test_acc)

  if epoch % 100 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
print("\n\n")


# plot decision boundary function ##################################
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits)).squeeze()  # binary

    # Reshape y_pred and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# plot decision boundary ######################################
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)

plt.show()

# plot loss ###################################
plt.plot(epoch_count, torch.tensor(train_loss_values).detach().numpy(), label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# plot accuracy ###############################
plt.plot(epoch_count, torch.tensor(train_acc_values).detach().numpy(), label="Train accuracy")
plt.plot(epoch_count, test_acc_values, label="Test accuracy")
plt.title("Training and test accuracy curves")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()

