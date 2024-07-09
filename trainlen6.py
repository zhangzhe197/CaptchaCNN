import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pyarrow.parquet as pq
from PIL import Image
import io,string
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision.models as models
from tqdm import tqdm


allchars = string.digits + string.ascii_letters + '#'
batch_size = 128
LR = 0.001
epoch = 50

class CNN(nn.Module):
    def __init__(self, num_classes=63 * 6):
        super(CNN, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the fully connected layer to match the number of classes in your specific task
        self.resnet18.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet18(x)


def makeLabel(labelStr, alphabet = allchars, num_class = 63):
    target = []
    for char in labelStr:
        vec = [0] * num_class
        vec[alphabet.find(char)] = 1
        target += vec
    return torch.Tensor(target)
def calculat_acc(output, target,num_classes = 63, num_char = 6):
    output, target = output.view(-1, num_classes), target.view(-1, num_classes)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, num_char), target.view(-1, num_char)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc


class CaptchaDataset(Dataset):
    def __init__(self, file_paths,transform):
        self.file_paths = file_paths
        self.transform = transform
        self.parquet_data = self.read_parquet_files()

    def read_parquet_files(self):
        # 从多个 Parquet 文件中读取数据，合并成一个数据帧
        dfs = [pq.read_table(file_path).to_pandas() for file_path in self.file_paths]
        return pd.concat(dfs, ignore_index=True)
    def __len__(self):
        return len(self.parquet_data)
    def __getitem__(self, idx):
        img_bytes = self.parquet_data.iloc[idx, 0]['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert('L') 
        img = self.transform(img)
        label = self.parquet_data.iloc[idx, 1].split("'")[1]
        if len(label) == 4:
            label = label + '##'
        elif len(label) == 5:
            label = label + '#'
        else:
            pass
        return img, makeLabel(label)

# 设置数据集文件路径
train_path = ['../Data/' + str(i) + 'train.parquet' for i in range(1,15)]
test_path = ['../Data/' + str(i) + 'test.parquet' for i in range(1,15)]

transform_train = transforms.Compose([
                                        transforms.Resize((50,160)),
                                        transforms.ToTensor(),  # 将PILImage转换为张量
                                      # 将[0,1]归一化到[-1,1]
                                        transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
                                        transforms.RandomRotation(10),
                                        transforms.Normalize([0.5],[0.5]),
                                      
                                    
                                      ])

transform_test = transforms.Compose([
                                       transforms.Resize((50,160)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5],[0.5])])

train_set = CaptchaDataset(train_path,transform_train)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=8)

test_set = CaptchaDataset(test_path,transform_test)
test_loader = DataLoader(test_set,batch_size= batch_size,num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = CNN().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
criterion = nn.MultiLabelSoftMarginLoss()
milestones = [15, 25,40] 
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

train_loss_history = []
test_loss_history = []
acc_history = []
best_acc = 0
def train(model = cnn, train_loader = train_loader,
          criterion = criterion,optimizer = optimizer,device =device,scheduler = scheduler):
        model.train()
        running_loss = 0
        for img, labels in tqdm(train_loader): 
            img, labels = img.to(device), labels.to(device)
            output = model(img)
            loss = criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        return running_loss / len(train_loader)

def test(model = cnn, loader = test_loader,
          criterion = criterion,device =device):
    cnn.eval()
    global best_acc
    with torch.no_grad():
        running_acc = 0
        running_loss = 0
        for img, labels in loader:
            img, labels = img.to(device), labels.to(device)
            output = model(img)
            loss = criterion(output, labels)
            running_loss += loss.item()
            running_acc += calculat_acc(output,labels)
        acc = running_acc / len(loader)
        if(acc > best_acc):
            torch.save(model.state_dict(),"Captcha.pth")
            best_acc = acc
            print("saved",end='\t')
        return running_loss / len(loader), acc


for i in range(epoch):
    trainLoss = train()
    testLoss, acc = test()
    train_loss_history.append(trainLoss)
    test_loss_history.append(testLoss)
    acc_history.append(acc)
    print(f"Epoch {i + 1}/{epoch}-> trainLoss = {trainLoss:.4f}, testLoss = {testLoss:.4f} ACC = {100 * acc:.2f}")

with open('training_history.txt', 'w') as file:
    file.write("Train Loss History:\n")
    for loss in train_loss_history:
        file.write(f"{loss}\n")

    file.write("\nTest Loss History:\n")
    for loss in test_loss_history:
        file.write(f"{loss}\n")

    file.write("\nAccuracy History:\n")
    for acc in acc_history:
        file.write(f"{acc}\n")

print("Data has been saved to 'training_history.txt'.")

