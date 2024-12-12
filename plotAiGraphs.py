import matplotlib.pyplot as plt
import os

# Define the path to the Downloads folder
def get_downloads_folder():
    return os.path.join(os.path.expanduser("~"), "Downloads")

# Data
titles = ["Training, Test, and Validation Accuracy for the Different Models", 
          "Train and Test Loss over Epochs", 
          "Train and Test Accuracy over Epochs"]
models = ["Logistic Regression", "XGBoost", "CNN"]
training_accuracies = [0.87, 0.81, 0.96] 
test_accuracies = [0.87, 0.81, 0.96]  

cnn_train_loss = [874.2468, 770.5683, 664.6612, 526.5851, 395.1995, 290.4222, 223.9882, 162.5824, 134.0541, 112.3830]
cnn_train_accuracy = [75.61, 77.16, 79.85, 84.05, 87.76, 91.07, 93.37, 95.34, 96.16, 96.83]
cnn_test_accuracy = [70.61, 72.05, 75.30, 80.12, 83.87, 86.93, 89.12, 91.67, 93.45, 95.58]
cnn_test_loss = [800.5683, 700.2345, 600.5621, 500.1235, 400.9834, 300.7654, 220.1234, 160.1235, 130.7654, 110.4567]

epochs = range(1, 11)
download_path = get_downloads_folder()

plt.figure(figsize=(10, 6))
plt.plot(models, training_accuracies, label="Training Accuracy", marker='o')
plt.plot(models, test_accuracies, label="Test Accuracy", marker='s')
plt.title(titles[0])
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.legend()
plt.grid()
graph_path = os.path.join(download_path, "training_test_validation_accuracy_line.png")
plt.savefig(graph_path)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs, cnn_train_loss, label='Train Loss', marker='o')
plt.plot(epochs, cnn_test_loss, label='Test Loss', marker='s')
plt.title(titles[1])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
graph_path = os.path.join(download_path, "train_test_loss_epochs.png")
plt.savefig(graph_path)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs, cnn_train_accuracy, label='Train Accuracy', marker='o')
plt.plot(epochs, cnn_test_accuracy, label='Test Accuracy', marker='s')
plt.title(titles[2])
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()
graph_path = os.path.join(download_path, "train_test_accuracy_epochs.png")
plt.savefig(graph_path)
plt.close()

print("Graphs saved to the Downloads folder.")
