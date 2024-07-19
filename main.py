import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

class CustomTransform(): # transform class
    def __call__(self, sample):
        X, y = sample
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y
    
class CustomDataset(Dataset): # dataset class
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = (self.X[idx], self.y[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample

# simple neural network to train data using nn.Module class
class MultiClassNet(nn.Module): 
    def __init__(self, input_size):
        super(MultiClassNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 5) # 5 classes

    def forward(self, x):
        # 1st layer
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 2nd layer
        out = self.fc2(out)
        out = self.relu(out)
        # 3rd layer
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        # output layer
        return self.fc4(out) # returns out

#import data directly from computer; TODO: copy path from your computer
df = pd.read_csv("/workspaces/AirQualityAI/air_quality_health_impact_data.csv")
df = df.drop(columns=['RecordID', 'RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions', 'HealthImpactScore']) # get rid of 'RecordID' and health metrics
df = pd.get_dummies(df, columns=['HealthImpactClass']).astype(float)
df.rename(columns={'HealthImpactClass_0.0':'HealthImpact_VeryHigh',
                   'HealthImpactClass_1.0':'HealthImpact_High',
                   'HealthImpactClass_2.0':'HealthImpact_Mod',
                   'HealthImpactClass_3.0':'HealthImpact_Low',
                   'HealthImpactClass_4.0':'HealthImpact_VeryLow'}, inplace=True)

# make y the target for classification of health impact
y = df[['HealthImpact_VeryHigh', 'HealthImpact_High', 'HealthImpact_Mod', 'HealthImpact_Low', 'HealthImpact_VeryLow']].values
X = df.drop(columns=['HealthImpact_VeryHigh', 'HealthImpact_High', 'HealthImpact_Mod', 'HealthImpact_Low', 'HealthImpact_VeryLow']).values

# standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split data into training, validation, and testing sets
transform = CustomTransform()
data = CustomDataset(X, y, transform=transform)

train_size = int(.6 * len(data)) # 60% training data
dev_size = int(.2 * len(data)) # 20% validation data
test_size = len(data) - train_size - dev_size

train_dataset, dev_dataset, test_dataset = random_split(data, [train_size, dev_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
input_size = X.shape[1]
model = MultiClassNet(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=.001)

train_accuracies = []
val_accuracies = []

# Training & Validation Loop
num_epochs = 20

for epoch in range(num_epochs):
    # training loop
    model.train()
    correct_train = 0
    total_train = 0
    for step, (inputs, labels) in enumerate(train_dataloader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == torch.argmax(labels,dim=1)).sum().item()

        if step % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
    
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    # validation loop
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0
    with torch.no_grad():
        for batch in dev_dataloader:
            inputs, labels = batch

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == torch.argmax(labels, dim=1)).sum().item()
    
    val_loss /= len(dev_dataloader)
    val_accuracy = 100 * correct_val / total_val
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Validations Loss: {val_loss:.4f}, Train Accuracy : {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

# plot training and validation accuracy over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Train and Validation Accuracy over Epochs')
plt.legend()
plt.show()

# Results 
model.eval() # set model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
    
    accuracy = correct / total * 100
    print(f'Test accuracy: {accuracy:.4f}%') # print final testing accuracy