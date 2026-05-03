import torch
import os
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from torch.optim import Adam, SGD
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

X = torch.load('data.pt')
Y = torch.load('target.pt')
indices = torch.randperm(len(X))
X, Y = X[indices], Y[indices]
X_train, X_val = X[:100], X[100:]
Y_train, Y_val = Y[:100], Y[100:]

CLASS_NAMES = ["setosa", "versicolor", "virginica"]
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainParams(BaseModel):
    hidden_neurons: int = 6
    epochs: int = 200
    lr: float = 0.01
    optimizer: str = "adam"

class PredictRequest(BaseModel):
    features: list[float]

@app.post("/train")
def train(params: TrainParams):
    model = Sequential(Linear(4, params.hidden_neurons), ReLU(), Linear(params.hidden_neurons, 3))
    loss_fn = CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=params.lr) if params.optimizer == "adam" else SGD(model.parameters(), lr=params.lr)
    losses, accuracies = [], []
    for epoch in range(params.epochs):
        y_hat = model(X_train)
        loss = loss_fn(y_hat, Y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == params.epochs - 1:
            with torch.no_grad():
                val_pred = model(X_val)
                acc = (val_pred.argmax(1) == Y_val).float().mean() * 100
                losses.append(loss.item())
                accuracies.append(acc.item())
    torch.save(model, "model.pt")
    with torch.no_grad():
        final_acc = (model(X_val).argmax(1) == Y_val).float().mean() * 100
    return {"final_accuracy": final_acc.item(), "loss_history": losses, "accuracy_history": accuracies}

@app.post("/predict")
def predict(request: PredictRequest):
    if not os.path.exists("model.pt"):
        return {"error": "Model not trained yet"}
    model = torch.load("model.pt", weights_only=False)#change
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([request.features]))
        pred = logits.argmax(1).item()
    return {"predicted_class": pred, "class_name": CLASS_NAMES[pred]}
