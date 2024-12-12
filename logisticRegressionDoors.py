import os
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk


def getInputPaths():
    root = tk.Tk()
    root.withdraw()
    print("Please select the folder containing the images.")
    imagesPath = filedialog.askdirectory(title="Select Images Folder")
    print("Please select the folder containing the labels.")
    labelsPath = filedialog.askdirectory(title="Select Labels Folder")
    return imagesPath, labelsPath


def get_downloads_folder():
    return os.path.join(os.path.expanduser("~"), "Downloads")

class Door:
    def __init__(self, box, category="GenericDoor"):
        self.box = box
        self.category = category

    def __repr__(self):
        return f"{self.category}(box={self.box})"

class RegularDoor(Door):
    def __init__(self, box):
        super().__init__(box, category="RegularDoor")

class CabinetDoor(Door):
    def __init__(self, box):
        super().__init__(box, category="CabinetDoor")

class RefrigeratorDoor(Door):
    def __init__(self, box):
        super().__init__(box, category="RefrigeratorDoor")

class Handle:
    def __init__(self, box):
        self.box = box

    def __repr__(self):
        return f"Handle(box={self.box})"

def save_evaluation_results(accuracy, report, output_file="evaluation_results.txt"):
    with open(output_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Evaluation results saved to {output_file}")

class MyLogisticRegression:
    def __init__(self, imagesPath, labelsPath):
        self.imagesPath = imagesPath
        self.labelsPath = labelsPath
        self.x = None
        self.y = None
        self.model = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.scaler_x = StandardScaler()

    def preprocess_images(self):
        image_files = [f for f in os.listdir(self.imagesPath) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"Total number of image files: {len(image_files)}")

        features = []
        targets = []

        for ifile in image_files:
            print(f"Processing image file: {ifile}")
            try:
                image = cv2.imread(os.path.join(self.imagesPath, ifile), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Skipping file due to loading error: {ifile}")
                    continue

                imageName, _ = os.path.splitext(ifile)
                lfile = os.path.join(self.labelsPath, f"{imageName}.txt")

                objects = []

                if os.path.exists(lfile) and os.path.getsize(lfile) > 0:
                    try:
                        with open(lfile, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    obj_class, x, y, w, h = map(float, parts)
                                    obj_class = int(obj_class)

                                    img_height, img_width = image.shape
                                    x_min = int((x - w / 2) * img_width)
                                    y_min = int((y - h / 2) * img_height)
                                    x_max = int((x + w / 2) * img_width)
                                    y_max = int((y + h / 2) * img_height)

                                    x_min = max(0, x_min)
                                    y_min = max(0, y_min)
                                    x_max = min(img_width, x_max)
                                    y_max = min(img_height, y_max)

                                    box = (x_min, y_min, x_max, y_max)

                                    if obj_class == 0:
                                        obj = RegularDoor(box)
                                    elif obj_class == 1:
                                        obj = Handle(box)
                                    elif obj_class == 2:
                                        obj = CabinetDoor(box)
                                    elif obj_class == 3:
                                        obj = RefrigeratorDoor(box)
                                    else:
                                        print(f"Unknown object class {obj_class}, skipping.")
                                        continue

                                    objects.append(obj)
                                    print(f"Parsed object: {obj}")

                        if objects:
                            for obj in objects:
                                x_min, y_min, x_max, y_max = obj.box
                                cropped_image = image[y_min:y_max, x_min:x_max]

                                if cropped_image.size == 0:
                                    print(f"Invalid cropped region for object: {obj}, skipping.")
                                    continue

                                cropped_image = cv2.resize(cropped_image, (64, 64))
                                cropped_image = cropped_image / 255.0

                                features.append(cropped_image.flatten())
                                targets.append(obj_class)
                    except Exception as e:
                        print(f"Error reading label file {lfile}: {e}")
                else:
                    print(f"Label file missing or empty for {ifile}, skipping.")

            except Exception as e:
                print(f"Error processing file {ifile}: {e}")
                continue

        self.x = np.array(features)
        self.y = np.array(targets)

        print(f"Total features extracted: {len(self.x)}")
        print(f"Total targets extracted: {len(self.y)}")

        if len(self.x) != len(self.y):
            raise ValueError(f"Mismatch between features ({len(self.x)}) and targets ({len(self.y)}).")

        valid_indices = self.y != -1
        self.x = self.x[valid_indices]
        self.y = self.y[valid_indices]

        print(f"Valid features: {len(self.x)}, Valid targets: {len(self.y)}")

        self.x = self.scaler_x.fit_transform(self.x)
        print("Features normalized.")

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42
        )
        print("Dataset split into training and testing sets.")


    def train_model(self):
        print("Training the logistic regression model...")
        self.model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        self.model.fit(self.x_train, self.y_train)
        print("Model training complete.")

    def evaluate_model(self, output_file="evaluation_results.txt"):
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model first.")
        print("Evaluating the logistic regression model...")

        y_pred = self.model.predict(self.x_test)

        valid_labels = [0, 1, 2, 3]

        invalid_test_labels = self.y_test[~np.isin(self.y_test, valid_labels)]
        if len(invalid_test_labels) > 0:
            print(f"Invalid labels found in y_test: {invalid_test_labels}")
        else:
            print("No invalid labels found in y_test.")

        invalid_pred_labels = y_pred[~np.isin(y_pred, valid_labels)]
        if len(invalid_pred_labels) > 0:
            print(f"Invalid labels found in y_pred: {invalid_pred_labels}")
        else:
            print("No invalid labels found in y_pred.")

        valid_indices = np.isin(self.y_test, valid_labels)
        y_test_filtered = self.y_test[valid_indices]
        y_pred_filtered = y_pred[valid_indices]

        accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
        report = classification_report(
            y_test_filtered,
            y_pred_filtered,
            labels=valid_labels,
            target_names=["RegularDoor", "Handle", "CabinetDoor", "RefrigeratorDoor"]
        )
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(report)

        downloads_path = get_downloads_folder()
        results_file = os.path.join(downloads_path, "evaluation_results.txt")
        with open(results_file, "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write("Classification Report:\n")
            f.write(report)
        print(f"Evaluation results saved to {results_file}")

        self.visualize_class_distribution()
        self.visualize_performance(y_test_filtered, y_pred_filtered)

    def visualize_class_distribution(self):
        valid_labels = [0, 1, 2, 3]
        valid_indices = np.isin(self.y, valid_labels)
        filtered_y = self.y[valid_indices]
        
        unique, counts = np.unique(filtered_y, return_counts=True)
        class_names = ["RegularDoor", "Handle", "CabinetDoor", "RefrigeratorDoor"]
        
        if len(unique) != len(class_names):
            print(f"Warning: Number of valid labels ({len(unique)}) does not match number of class names ({len(class_names)}).")
        
        plt.bar(class_names[:len(unique)], counts, color='skyblue')
        plt.title("Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        plot_file = os.path.join(downloads_path, "class_distribution.png")
        plt.savefig(plot_file)
        plt.show()
        
        print(f"Class distribution plot saved to {plot_file}")

    def visualize_performance(self, y_true, y_pred):
        class_names = ["RegularDoor", "Handle", "CabinetDoor", "RefrigeratorDoor"]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
        )

        bar_width = 0.25
        r1 = np.arange(len(class_names))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]

        downloads_path = get_downloads_folder()
        plot_file = os.path.join(downloads_path, "performance_metrics.png")

        plt.bar(r1, precision, color='b', width=bar_width, edgecolor='grey', label='Precision')
        plt.bar(r2, recall, color='g', width=bar_width, edgecolor='grey', label='Recall')
        plt.bar(r3, f1, color='r', width=bar_width, edgecolor='grey', label='F1 Score')

        plt.xlabel('Classes', fontweight='bold')
        plt.xticks([r + bar_width for r in range(len(class_names))], class_names)
        plt.title("Model Performance Metrics")
        plt.legend()

        plt.savefig(plot_file)
        plt.show()

        print(f"Performance metrics plot saved to {plot_file}")

    def visualize_training_vs_test_accuracy(self):
        training_accuracy = self.model.score(self.x_train, self.y_train)
        test_accuracy = self.model.score(self.x_test, self.y_test)

        metrics = ["Training Accuracy", "Test Accuracy"]
        values = [training_accuracy, test_accuracy]

        downloads_path = get_downloads_folder()
        plot_file = os.path.join(downloads_path, "training_vs_test_accuracy.png")

        plt.bar(metrics, values, color='purple', alpha=0.7, edgecolor='black')
        plt.title("Training vs Test Accuracy")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)

        plt.savefig(plot_file)
        plt.show()

        print(f"Training vs Test Accuracy plot saved to {plot_file}")




if __name__ == '__main__':
    imagesPath, labelsPath = getInputPaths()

    classifier = MyLogisticRegression(imagesPath, labelsPath)
    classifier.preprocess_images()
    classifier.train_model()
    classifier.evaluate_model()
    classifier.visualize_training_vs_test_accuracy()
