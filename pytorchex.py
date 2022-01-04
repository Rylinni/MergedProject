import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import pickle

actual_data = pickle.load(open("training/train_4129.sav", 'rb'))
x = torch.tensor(actual_data[0]).float()
y = torch.reshape(torch.tensor(actual_data[1]), (-1,1)).float()

x, y = Variable(x), Variable(y)

torch_dataset = Data.TensorDataset(x, y)
torch_dataloader = Data.DataLoader(torch_dataset, batch_size=64, shuffle=True)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(5*5*8, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
        )

    def forward(self, x):
        prediction = self.linear_relu_stack(x)
        return prediction

model = NeuralNetwork().to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(torch_dataloader, model, loss_func, optimizer)
    test(torch_dataloader, model, loss_func)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

train(torch_dataloader, model, loss_func, optimizer)

actual_data = pickle.load(open("training/train_4129.sav", 'rb'))
test_x = actual_data[0][0]
print(test_x)
res = model(torch.tensor(test_x).float())[0].item()
print(res)
res = model(torch.tensor(actual_data[0]).float())[0]
print(res)