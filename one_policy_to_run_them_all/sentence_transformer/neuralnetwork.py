import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# X shape: (num_samples, input_size)
# Y shape: (num_samples, output_size)

class NeuralNetwork():
    def __init__(self, inputSize, outputSize):
        self.input_size = inputSize
        self.output_size = outputSize
        self.loss_fn = nn.MSELoss()
        self.lr=0.001

        self.createModel()
    
    def setTrainSet(self, X, y):
        X = X.float()
        y = y.float()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        return loader

    def createModel(self):
        model = nn.Sequential(
            nn.Linear(self.input_size, 300),
            nn.Sigmoid(),
            nn.Linear(300, 200),
            nn.Sigmoid(),
            nn.Linear(200, 100),
            nn.Sigmoid(),
            nn.Linear(100, 50),
            nn.Sigmoid(),
            nn.Linear(50, self.output_size)
        )
        self.model = model
    
    def trainModel(self, X, y, epochs = 1000):

        self.loader = self.setTrainSet(X,y)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            total_loss=0

            for X_batch, y_batch in self.loader:
                output = self.model(X_batch)
                loss = self.loss_fn(output, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            if (epoch+1)%100 == 0:
                avg_loss = total_loss/len(self.loader)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


    def testModel(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_test)

            loss = self.loss_fn(output, y_test)

        print("Test Loss:", loss.item())

    def saveModel(model: NeuralNetwork, name):
        torch.save(model.model.state_dict(), name)
        
    def loadModel(input_size, output_size, filename):

        if input_size == input_size and output_size == output_size:
            loaded_model = NeuralNetwork(input_size, output_size)
            loaded_model.model.load_state_dict(torch.load(filename))
            return loaded_model
        else:
            pass