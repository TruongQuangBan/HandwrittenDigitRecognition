import os
import time

from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib


MNIST_CACHE_FILE = "mnist_784.pkl"
MODEL_FILE = "model.pkl"
OPENML_CACHE_DIR = "openml_cache"
IMAGE_DIR = "images"
REPORT_FILE = "report.txt"


def load_mnist():
    try:
        mnist = joblib.load(MNIST_CACHE_FILE)
        print(f"Da tai MNIST tu cache cuc bo: {MNIST_CACHE_FILE}")
        return mnist["data"], mnist["target"]
    except FileNotFoundError:
        print("Chua co cache cuc bo. Dang tai MNIST tu OpenML...")

    mnist = fetch_openml(
        "mnist_784",
        version=1,
        cache=True,
        as_frame=False,
        data_home=OPENML_CACHE_DIR,
        parser="liac-arff",
    )
    X = mnist.data
    y = mnist.target.astype(int)
    joblib.dump({"data": X, "target": y}, MNIST_CACHE_FILE)
    print(f"Da luu cache MNIST vao file {MNIST_CACHE_FILE}")
    return X, y


def save_sample_images(X_train, y_train):
    os.makedirs(IMAGE_DIR, exist_ok=True)

    sample_images = []
    for index in range(5):
        image_array = X_train[index].reshape(28, 28).astype("uint8")
        image = Image.fromarray(image_array)
        label = int(y_train[index])
        image.save(os.path.join(IMAGE_DIR, f"train_sample_{index}_label_{label}.png"))
        sample_images.append(image)

    montage = Image.new("L", (28 * len(sample_images), 28))
    for index, image in enumerate(sample_images):
        montage.paste(image, (index * 28, 0))
    montage.save(os.path.join(IMAGE_DIR, "mnist_samples_montage.png"))
    print(f"Da luu 5 anh mau va anh ghep vao thu muc {IMAGE_DIR}/")


def write_report(train_count, test_count, training_time, accuracy):
    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write("Bao cao huan luyen MNIST\n")
        report.write("========================\n")
        report.write(f"So luong train: {train_count}\n")
        report.write(f"So luong test: {test_count}\n")
        report.write(f"Thoi gian huan luyen: {training_time:.2f} giay\n")
        report.write(f"Do chinh xac tren tap test: {accuracy:.4f}\n")
        report.write("\n")
        report.write(
            "Ghi chu: Logistic Regression one-vs-rest huan luyen mot bo phan "
            "loai nhi phan cho moi chu so, sau do chon lop co diem du doan cao nhat.\n"
        )
    print(f"Da ghi bao cao vao file {REPORT_FILE}")


def main():
    X, y = load_mnist()

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print(f"So luong train: {len(X_train)}")
    print(f"So luong test: {len(X_test)}")
    save_sample_images(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(multi_class="ovr", max_iter=100, n_jobs=1)

    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    print(f"Thoi gian huan luyen: {training_time:.2f} giay")

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Do chinh xac tren tap test: {accuracy:.4f}")

    joblib.dump({"scaler": scaler, "model": model}, MODEL_FILE)
    print(f"Da luu model thanh cong vao file {MODEL_FILE}")
    write_report(len(X_train), len(X_test), training_time, accuracy)


if __name__ == "__main__":
    main()
