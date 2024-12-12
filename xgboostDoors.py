import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import tkinter as tk
from tkinter import filedialog

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

class MyXGBoostModel:
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

                                cropped_image = cv2.resize(cropped_image, (64, 64))
                                cropped_image = cropped_image / 255.0

                                features.append(cropped_image.flatten())
                                targets.append(obj_class)
                        else:
                            targets.append(-1)
                    except Exception as e:
                        print(f"Error reading label file {lfile}: {e}")
                        targets.append(-1)
                else:
                    print(f"Label file missing or empty for {ifile}, assigning -1")
                    targets.append(-1)

            except Exception as e:
                print(f"Error processing file {ifile}: {e}")
                continue

        self.x = np.array(features)
        self.y = np.array(targets)

        print(f"Total features extracted: {len(features)}")
        print(f"Total targets extracted: {len(targets)}")

        if len(features) != len(targets):
            print("Warning: Mismatch between features and targets. Adjusting...")
            min_length = min(len(features), len(targets))
            self.x = self.x[:min_length]
            self.y = self.y[:min_length]

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
        print("Training the XGBoost model...")
        self.model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            objective="multi:softmax",
            num_class=4,
            eval_metric="mlogloss",
            use_label_encoder=False
        )
        self.model.fit(self.x_train, self.y_train)
        print("Model training complete.")

    def evaluate_model(self):
        print("Evaluating the XGBoost model...")

        y_pred = self.model.predict(self.x_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, labels=[0, 1, 2, 3], zero_division=0
        )

        report = classification_report(
            self.y_test,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=["RegularDoor", "Handle", "CabinetDoor", "RefrigeratorDoor"]
        )

        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(report)

        downloads_path = get_downloads_folder()
        results_file = os.path.join(downloads_path, "xgboost_evaluation_results.txt")
        save_evaluation_results(accuracy, report, results_file)

        self.metrics = {
            "accuracy": accuracy,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1_score": f1.tolist()
        }
        print("Metrics for final project report:")
        print(self.metrics)

        csv_file = os.path.join(downloads_path, "xgboost_metrics.csv")
        with open(csv_file, "w") as f:
            f.write("Class,Precision,Recall,F1-Score\n")
            for i, class_name in enumerate(["RegularDoor", "Handle", "CabinetDoor", "RefrigeratorDoor"]):
                f.write(f"{class_name},{precision[i]:.4f},{recall[i]:.4f},{f1[i]:.4f}\n")
        print(f"Class-wise metrics saved to {csv_file}")

        return accuracy, precision, recall, f1

    def visualize_class_distribution(self):
        unique, counts = np.unique(self.y, return_counts=True)
        plt.bar(["RegularDoor", "Handle", "CabinetDoor", "RefrigeratorDoor"], counts, color="skyblue")
        plt.title("Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.show()

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
        plot_file = os.path.join(downloads_path, "xgboost_performance_metrics.png")

        plt.bar(r1, precision, color='b', width=bar_width, edgecolor='grey', label='Precision')
        plt.bar(r2, recall, color='g', width=bar_width, edgecolor='grey', label='Recall')
        plt.bar(r3, f1, color='r', width=bar_width, edgecolor='grey', label='F1 Score')

        plt.xlabel('Classes', fontweight='bold')
        plt.xticks([r + bar_width for r in range(len(class_names))], class_names)
        plt.title("XGBoost Model Performance Metrics")
        plt.legend()

        plt.savefig(plot_file)
        plt.show()

    def visualize_training_vs_test_accuracy(self):
        training_accuracy = self.model.score(self.x_train, self.y_train)
        test_accuracy = self.model.score(self.x_test, self.y_test)

        metrics = ["Training Accuracy", "Test Accuracy"]
        values = [training_accuracy, test_accuracy]

        downloads_path = get_downloads_folder()
        plot_file = os.path.join(downloads_path, "xgboost_training_vs_test_accuracy.png")

        plt.bar(metrics, values, color='purple', alpha=0.7, edgecolor='black')
        plt.title("Training vs Test Accuracy")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)

        plt.savefig(plot_file)
        plt.show()

        print(f"Training vs Test Accuracy plot saved to {plot_file}")

if __name__ == "__main__":
    imagesPath, labelsPath = getInputPaths()

    classifier = MyXGBoostModel(imagesPath, labelsPath)
    classifier.preprocess_images()
    classifier.train_model()

    accuracy, precision, recall, f1 = classifier.evaluate_model()

    classifier.visualize_training_vs_test_accuracy()
    classifier.visualize_class_distribution()

    print("\nSummary:")
    print(f"Accuracy: {accuracy:.4f}")
    for i, class_name in enumerate(["RegularDoor", "Handle", "CabinetDoor", "RefrigeratorDoor"]):
        print(f"{class_name} -> Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-Score: {f1[i]:.4f}")

