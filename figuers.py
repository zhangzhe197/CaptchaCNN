import matplotlib.pyplot as plt

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file.readlines()]
    return data

def getInfo(file_path):
    all_data = read_data(file_path)

    # 分离训练损失、测试损失和准确度的数据
    train_loss_history = all_data[:len(all_data)//3]
    test_loss_history = all_data[len(all_data)//3:2*len(all_data)//3]
    accuracy_history = all_data[2*len(all_data)//3:]
    return train_loss_history, test_loss_history, accuracy_history

file_path = "training_history.txt"
Res34LossTrain, Res34LossTest, Res34Acc = getInfo("training_historyRes34.txt")
train_loss_history, test_loss_history,accuracy_history = getInfo(file_path=file_path)
# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 12))

# Plot train and test loss in the first subplot
ax1.plot(train_loss_history, label='Res18 Net Train Loss')
ax1.plot(test_loss_history, label='Res18 Net Test Loss')
ax1.plot(Res34LossTrain, label='Res34 Net Train Loss')
ax1.plot(Res34LossTest, label='Res34 Net Test Loss')
ax1.set_title('Train and Test Loss History')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot accuracy in the second subplot
ax2.plot(accuracy_history, label='Res18 Net Accuracy')
ax2.plot(Res34Acc, label='Res34 Net Accuracy')
ax2.set_title('Accuracy History')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

