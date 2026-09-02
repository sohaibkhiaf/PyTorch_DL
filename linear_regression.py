import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path


print("Version ============================================")
print(f"Torch version: {torch.__version__}")
print("\n\n")


print("Device agnostic code ===================================")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("\n\n")


print(f"Loading data =========================================")
# load data
df = pd.read_csv('ames_housing.csv')

# predict house sale price based on ground living area only
X= df["GrLivArea"]
y= df["SalePrice"]

# convert array to tensor
X= torch.tensor(X).unsqueeze(dim=1).type(torch.float)
y= torch.tensor(y).unsqueeze(dim=1).type(torch.float)

print(f"X: {X}")
print(f"X.shape: {X.shape}")
print(f"y: {y}")
print(f"y.shape: {X.shape}")
print("\n\n")

print(f"Train- test split =====================================")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train Shape: {X_train.shape}")
print(f"y_train Shape: {y_train.shape}")
print(f"X_test Shape: {X_test.shape}")
print(f"y_test Shape: {y_test.shape}")
print("\n\n")


# plot data function #########################################
def plot_predictions(train_features=X_train,
                    train_labels=y_train,
                    test_features=X_test,
                    test_labels=y_test,
                    predictions=None):
  plt.figure(figsize=(10, 7))
  plt.scatter(train_features, train_labels, c="b", s=4, label="Training data")
  plt.scatter(test_features, test_labels, c="g", s=4, label="Testing data")
  if predictions is not None:
    plt.scatter(test_features, predictions, c="r", s=4, label="Predictions")
  plt.legend(prop={"size": 14})
  plt.show()

# plot predictions ####################################
plot_predictions(X_train, y_train, X_test, y_test)

# create a linear model by subclassing nn.Module
class LinearRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()

    # initialize model layers (built-in nn.Linear)
    self.linear_layer = nn.Linear(in_features=1, out_features=1)

  # forward() defines the computation in the model
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x)


# create a custom linear model by subclassing nn.Module
class LinearRegressionModelCustom(nn.Module):
  def __init__(self):
    super().__init__()

    # initialize model parameters (custom)
    self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

  # forward() defines the computation in the model
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.weight * x + self.bias


print(f"Create model ======================================= ")
# set manual seed
torch.manual_seed(42)

# create model
model = LinearRegressionModel()

# set the model to use the target device
model.to(device)

print(f"State dict: {model.state_dict()}")
print(f"Model on device: {next(model.parameters()).device}")
print("\n\n")


print(f"Create loss function and optimizer =========================")
# setup loss function and optimizer
loss_fn = nn.L1Loss() # same as MAE (Mean Absolute Error)
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01)

print(f"Loss function: {loss_fn}")
print(f"Optimizer: {optimizer}")
print("\n\n")


print("Training loop ==================================")
# training loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 200

# move data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
  # put the model in the training mode
  model.train()

  # forward pass
  y_pred = model(X_train)

  # calculate the loss
  loss = loss_fn(y_pred, y_train)

  # zero grad (clear old gradients)
  optimizer.zero_grad()

  # perform backpropagation (calculates gradients)
  loss.backward()

  # optimizer step updates the parameters
  # using SGD (Stochastic Gradient Descent)
  optimizer.step()

  # testing
  model.eval()
  with torch.inference_mode():
    y_pred = model(X_test)

    test_loss = loss_fn(y_pred, y_test)

  # logging
  if epoch % 10 == 0:
    epoch_count.append(epoch)
    loss_values.append(loss.detach().item())
    test_loss_values.append(test_loss.detach().item())
    print(f"Epoch: {epoch} | Loss: {loss:.2f} | Test loss: {test_loss:.2f}")
    print(f"Model state dict: {model.state_dict()}")
print("\n\n")

# plot predictions  ####################################
model.eval()
with torch.inference_mode():
  y_pred = model(X_test)
plot_predictions(predictions=y_pred.cpu())

# plot loss ############################################
plt.plot(epoch_count, loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

print("Saving model ================================")
# create checkpoints directory
MODEL_PATH = Path("checkpoints")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# create model save path
MODEL_NAME = "model_state_dict.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# save the model state dict
print(f"Saving model in: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
print("\n\n")



print("Load saved model ============================")
# load saved model state dict
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model.to(device)

# check model device + state dict
print(f"Model on device: {next(loaded_model.parameters()).device}")
print(f"Model state dict: {loaded_model.state_dict()}")
print("\n\n")


