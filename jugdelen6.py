import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pyarrow.parquet as pq
from PIL import Image
import io,string
import torch.nn as nn
from torch.optim import lr_scheduler
import pandas as pd
import torchvision.models as models

allchars = string.digits + string.ascii_letters + '#'
batch_size = 1
LR = 0.001
epoch = 50
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
def extract_predicted_string(output, allchars=allchars, num_char=6):
    # Assuming len(allchars) is the total number of classes
    num_classes = len(allchars)
    
    # Rest of the function remains unchanged
    output = output.view(-1, num_classes)
    
    output = nn.functional.softmax(output, dim=1)
    
    output = torch.argmax(output, dim=1)
    
    output = output.view(-1, num_char)
    
    predicted_strings = []
    for sample_output in output:
        predicted_string = ''.join([allchars[idx] for idx in sample_output])
        predicted_strings.append(predicted_string.replace('#',''))
    
    return predicted_strings


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
train_path = ['Data/' + str(i) + 'train.parquet' for i in range(1,15)]
test_path = ['Data/' + str(i) + 'test.parquet' for i in range(1,15)]

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

test_set = CaptchaDataset(test_path,transform_test)
test_loader = DataLoader(test_set,batch_size= batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import numpy as np

# 定义绘制图表的函数
def plot_images_with_labels(images, true_labels, predicted_labels, num_images=64):
    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    axes = axes.ravel()

    for i in np.arange(0, num_images):
        # Original label
        true_label = true_labels[i]
        
        # Predicted label
        predicted_label = predicted_labels[i]
        
        # Mark incorrect predictions in red
        color = 'black' if true_label == predicted_label else 'red'
        img = Image.open(io.BytesIO(images[i]['bytes'])).convert('L') 
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}', color=color)
        axes[i].axis('off')

    plt.subplots_adjust(wspace=1)

# 加载模型
model = CNN()
model.load_state_dict(torch.load('Captcha.pth'))  # 替换为你的模型路径
model.to(device)
model.eval()

predictions = []
true_labels = []
random_indices = np.random.choice(len(test_set), size=64, replace=False)



with torch.no_grad():
    for idx in random_indices:
        images, labels = test_set[idx]
        images, labels = images.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(images)
        predicted_strings = extract_predicted_string(outputs)
        predictions.extend(predicted_strings)
        true_labels.extend(extract_predicted_string(labels))

    plot_images_with_labels([test_set.parquet_data["image"][idx] for idx in random_indices], true_labels, predictions)

    plt.show()