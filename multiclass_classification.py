import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np


print("Version =========================================")
print(f"Torch version: {torch.__version__}")
print("\n\n")


print("Device =========================================")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("\n\n")



# create multi-class data #############################
X, y = make_blobs(n_samples=1000,
                            n_features=2,
                            centers=4,
                            cluster_std=1.5,
                            random_state=42)

# turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.LongTensor)

# plot data
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()


print("Train- test split ===============================")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print("\n\n")



# build a multi-class classification model #################################
class BlobModel(nn.Module):
  def __init__(self, input_features, output_features, hidden_units=8):
    super().__init__()
    self.linear_layer_stack = nn.Sequential(
        nn.Linear(in_features=input_features, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=output_features),
    )

  def forward(self, x):
    return self.linear_layer_stack(x)

print("Classes ==========================================")
print(f"Classes: {torch.unique(y_train)}")
print("\n\n")



print("Create model ==========================================")
model = BlobModel(input_features=2, # features
                  output_features=4, # number of classes
                  hidden_units=8).to(device)
print(f"Model: {model}")
print("\n\n")



# create accuracy function ################################
def accuracy_fn(y_true, y_pred):
  correct = 0
  for i in range(len(y_pred)):
    if y_pred[i].item() == y_true[i].item():
      correct+= 1
  acc = (correct / len(y_pred)) * 100
  return acc

# loss function and optimizer ###############################
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)

print("Evaluate model before training =================================")
model.eval()
with torch.inference_mode():
  y_logit = model(X_test.to(device))
y_prob = torch.softmax(y_logit, dim=1)
y_pred = torch.argmax(y_prob, dim=1)

print(f"Logits: {y_logit[:10]}")
print(f"Probs: {y_prob[:10]}")
print(f"Pred: {y_pred[:10]}")
print(f"Actual: {y_test[:10]}")
print(f"Accuracy(10 samples): {accuracy_fn(y_true=y_test[:10], y_pred=y_pred[:10])}%")
print("\n\n")



print("Training loop ======================================")
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# set number of epochs
epochs = 100

# put data to the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# loss and acc values
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
  y_logit = model(X_train)
  y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

  # calculate loss
  loss = loss_fn(y_logit, y_train)
  acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
  train_loss_values.append(loss)
  train_acc_values.append(acc)

  # reset optimizer
  optimizer.zero_grad()

  # backpropagation
  loss.backward()

  # optimizer step
  optimizer.step()

  model.eval()
  with torch.inference_mode():
    test_logit = model(X_test)
    test_pred = torch.softmax(test_logit, dim=1).argmax(dim=1)

    test_loss = loss_fn(test_logit, y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    test_loss_values.append(test_loss)
    test_acc_values.append(test_acc)

  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
print("\n\n")



# plot decision boundary function #######################################
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
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# plot data+ decision boundary #########################################
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)

plt.show()

print("Making predictions (after training ) =================================")
model.eval()
with torch.inference_mode():
  y_logit = model(X_test.to(device))
y_prob = torch.softmax(y_logit, dim=1)
y_pred = torch.argmax(y_prob, dim=1)

print(f"Logits: {y_logit[:10]}")
print(f"Probs: {y_prob[:10]}")
print(f"Pred: {y_pred[:10]}")
print(f"Actual: {y_test[:10]}")
print(f"Accuracy(10 samples): {accuracy_fn(y_true=y_test[:10], y_pred=y_pred[:10])}%")
print("\n\n")


