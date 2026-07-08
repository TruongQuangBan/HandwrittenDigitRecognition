# Báo cáo đề tài: Xây dựng hệ thống nhận diện chữ số viết tay sử dụng Logistic Regression, Flask và Docker

## Thông tin chung

**Tên dự án:** Handwritten Digit Recognition  
**Kho mã nguồn:** <https://github.com/TruongQuangBan/HandwrittenDigitRecognition.git>  
**URL demo cloud:** <!-- URL_DEMO -->

## Giới thiệu

Nhận diện chữ số viết tay là một bài toán cơ bản nhưng có ý nghĩa quan trọng trong lĩnh vực học máy và thị giác máy tính. Bài toán này yêu cầu hệ thống tự động phân loại một ảnh chữ số viết tay vào một trong mười lớp từ `0` đến `9`. Đây là nền tảng cho nhiều ứng dụng thực tế như số hóa biểu mẫu, nhận diện mã bưu chính, xử lý phiếu khảo sát và các hệ thống nhập liệu tự động.

Trong đề tài này, hệ thống được xây dựng theo hướng hoàn chỉnh từ huấn luyện mô hình đến triển khai ứng dụng web. Mô hình học máy được huấn luyện trên tập dữ liệu MNIST, sau đó được lưu lại để sử dụng trong một API Flask. Người dùng có thể vẽ chữ số trực tiếp trên giao diện web bằng Canvas; dữ liệu ảnh sẽ được chuyển thành mảng pixel, gửi đến server, xử lý qua mô hình và trả về kết quả dự đoán dưới dạng JSON.

Mục tiêu chính của đề tài gồm:

- Huấn luyện mô hình phân loại chữ số viết tay từ ảnh grayscale kích thước `28x28`.
- Xây dựng API Flask để nhận dữ liệu ảnh và trả về kết quả dự đoán.
- Xây dựng giao diện web sử dụng Canvas để người dùng vẽ chữ số.
- Đóng gói ứng dụng bằng Docker để thuận tiện triển khai trên môi trường khác hoặc cloud.

Phạm vi đề tài tập trung vào tập dữ liệu MNIST với `60.000` ảnh huấn luyện và `10.000` ảnh kiểm thử. Thuật toán được sử dụng là Logistic Regression theo chiến lược one-vs-rest. Kết quả thực nghiệm hiện tại đạt độ chính xác `0.9175` trên tập kiểm thử.

## Chương 1: Tổng quan hệ thống

### 1.1. Bài toán nhận diện chữ số viết tay

Bài toán nhận diện chữ số viết tay có đầu vào là một ảnh chứa một chữ số viết tay và đầu ra là nhãn tương ứng trong tập `{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}`. Trong dự án này, ảnh đầu vào được quy chuẩn về dạng ảnh grayscale kích thước `28x28`, tương ứng với `784` giá trị pixel.

Với dữ liệu MNIST, mỗi ảnh đã được chuẩn hóa theo định dạng cố định. Tuy nhiên, trong môi trường thực tế, người dùng vẽ chữ số trên Canvas nên dữ liệu có thể khác với ảnh trong tập huấn luyện về vị trí, kích thước nét vẽ, độ dày nét và phong cách viết. Vì vậy, hệ thống cần có bước chuyển đổi ảnh Canvas về cùng định dạng mà mô hình đã học.

### 1.2. Kiến trúc tổng thể

Hệ thống được xây dựng theo mô hình client-server:

- **Client:** file `static/index.html`, gồm HTML, CSS và JavaScript. Thành phần chính là Canvas cho phép người dùng vẽ chữ số.
- **Server:** file `app.py`, sử dụng Flask để phục vụ giao diện và cung cấp API dự đoán.
- **Model:** file `model.pkl`, chứa `StandardScaler` và mô hình `LogisticRegression` đã huấn luyện.
- **Training pipeline:** file `train.py`, chịu trách nhiệm tải dữ liệu MNIST, tiền xử lý, huấn luyện, đánh giá và lưu mô hình.
- **Docker:** file `Dockerfile`, định nghĩa môi trường chạy ứng dụng trong container.

Luồng xử lý dự đoán của hệ thống:

1. Người dùng vẽ chữ số trên Canvas trong trình duyệt.
2. JavaScript thu nhỏ ảnh Canvas từ `280x280` về `28x28`.
3. Ảnh `28x28` được chuyển thành mảng `784` giá trị grayscale.
4. Client gửi request bằng Fetch API đến endpoint `/predict`.
5. Flask nhận JSON, kiểm tra dữ liệu đầu vào và chuyển mảng thành `numpy.ndarray`.
6. Dữ liệu được chuẩn hóa bằng `StandardScaler` đã lưu trong `model.pkl`.
7. Mô hình Logistic Regression dự đoán chữ số.
8. Flask trả kết quả về client dưới dạng JSON.
9. Giao diện hiển thị chữ số được dự đoán.

Tóm tắt kiến trúc:

```text
Canvas -> JavaScript xử lý ảnh -> Fetch API -> Flask API -> StandardScaler -> LogisticRegression -> JSON response
```

### 1.3. Cấu trúc mã nguồn

```text
.
├── Dockerfile
├── app.py
├── images/
│   ├── mnist_samples_montage.png
│   ├── train_sample_0_label_5.png
│   ├── train_sample_1_label_0.png
│   ├── train_sample_2_label_4.png
│   ├── train_sample_3_label_1.png
│   └── train_sample_4_label_9.png
├── mnist_784.pkl
├── model.pkl
├── openml_cache/
├── report.txt
├── requirements.txt
├── static/
│   └── index.html
└── train.py
```

Các file chính trong dự án:

- `train.py`: script huấn luyện mô hình trên tập MNIST.
- `app.py`: ứng dụng Flask, cung cấp giao diện web và API dự đoán.
- `static/index.html`: giao diện vẽ chữ số bằng Canvas.
- `model.pkl`: mô hình đã huấn luyện, gồm scaler và classifier.
- `report.txt`: kết quả huấn luyện gần nhất.
- `Dockerfile`: cấu hình đóng gói ứng dụng bằng Docker.
- `requirements.txt`: danh sách thư viện Python cần cài đặt.

## Chương 2: Cơ sở lý thuyết

### 2.1. Tập dữ liệu MNIST

MNIST là tập dữ liệu phổ biến trong các bài toán nhận diện chữ số viết tay. Tập dữ liệu gồm ảnh grayscale của các chữ số từ `0` đến `9`. Trong dự án này, dữ liệu được lấy thông qua `fetch_openml("mnist_784", version=1)` của scikit-learn và được cache cục bộ vào file `mnist_784.pkl`.

Quy mô dữ liệu sử dụng:

| Tập dữ liệu | Số lượng ảnh |
|---|---:|
| Train | 60.000 |
| Test | 10.000 |
| Tổng cộng | 70.000 |

Mỗi ảnh có kích thước `28x28` pixel. Khi biểu diễn dưới dạng vector, mỗi ảnh có `28 * 28 = 784` đặc trưng.

### 2.2. Tiền xử lý ảnh: grayscale, resize và flatten

Trong hệ thống, dữ liệu ảnh được chuẩn hóa về cùng định dạng trước khi đưa vào mô hình. Quá trình tiền xử lý gồm các bước chính sau.

#### 2.2.1. Chuyển ảnh về grayscale

Đối với endpoint `/predict_base64`, ảnh đầu vào có thể là ảnh màu hoặc ảnh có nhiều kênh. Hàm `image_to_model_input` trong `app.py` chuyển ảnh về grayscale bằng:

```python
image = image.convert("L")
```

Chế độ `"L"` của Pillow biểu diễn ảnh grayscale, trong đó mỗi pixel là một giá trị cường độ sáng. Việc chuyển về grayscale giúp dữ liệu tương thích với MNIST, vì MNIST cũng là tập ảnh một kênh.

#### 2.2.2. Resize ảnh về 28x28

Ảnh đầu vào được resize về kích thước `28x28`:

```python
image = image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
```

Kích thước này là định dạng chuẩn của MNIST. Nếu ảnh đầu vào không được resize, số lượng đặc trưng sẽ không khớp với mô hình đã huấn luyện.

Trên giao diện web, Canvas chính có kích thước `280x280` để người dùng dễ vẽ. Trước khi gửi lên server, JavaScript tạo một Canvas nhỏ kích thước `28x28` và vẽ lại nội dung từ Canvas lớn:

```javascript
smallCanvas.width = 28;
smallCanvas.height = 28;
smallCtx.drawImage(canvas, 0, 0, 28, 28);
```

#### 2.2.3. Flatten ảnh thành vector 784 chiều

Mô hình Logistic Regression trong scikit-learn nhận dữ liệu đầu vào dạng bảng hai chiều, trong đó mỗi dòng là một mẫu và mỗi cột là một đặc trưng. Vì vậy, ảnh `28x28` cần được chuyển thành vector một chiều gồm `784` phần tử.

Trong `train.py`, mỗi mẫu MNIST vốn đã ở dạng `784` đặc trưng khi lấy từ OpenML với tên `mnist_784`. Trong `app.py`, ảnh base64 sau khi resize được chuyển thành mảng NumPy và flatten:

```python
return np.asarray(image, dtype=float).reshape(-1)
```

Trên endpoint `/predict`, server yêu cầu trường `image` là một mảng có đúng `784` giá trị:

```python
if not isinstance(image, list) or len(image) != 784:
    return jsonify({"error": "Field 'image' must be an array of 784 numbers."}), 400
```

Việc flatten giúp biến dữ liệu ảnh thành vector đặc trưng:

```text
Ảnh 28x28 -> ma trận 28 hàng, 28 cột -> vector 784 chiều
```

Mỗi phần tử trong vector biểu diễn cường độ sáng của một pixel.

### 2.3. Chuẩn hóa dữ liệu bằng StandardScaler

Sau khi dữ liệu ảnh được biểu diễn dưới dạng vector, dự án sử dụng `StandardScaler` để chuẩn hóa các đặc trưng:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

`StandardScaler` biến đổi mỗi đặc trưng theo công thức:

```text
z = (x - mean) / standard_deviation
```

Trong đó:

- `x` là giá trị gốc của đặc trưng.
- `mean` là giá trị trung bình của đặc trưng trên tập huấn luyện.
- `standard_deviation` là độ lệch chuẩn của đặc trưng trên tập huấn luyện.
- `z` là giá trị sau chuẩn hóa.

Việc chuẩn hóa có vai trò quan trọng vì Logistic Regression là mô hình tối ưu hóa dựa trên các trọng số của đặc trưng. Nếu các đặc trưng có thang đo khác nhau, quá trình tối ưu có thể hội tụ chậm hoặc không ổn định. Với dữ liệu ảnh, các pixel có giá trị trong khoảng `0` đến `255`; chuẩn hóa giúp đưa các đặc trưng về miền giá trị phù hợp hơn cho quá trình học.

Một nguyên tắc quan trọng là `StandardScaler` chỉ được `fit` trên tập train. Sau đó, cùng scaler này được dùng để `transform` tập test và dữ liệu dự đoán thực tế. Điều này tránh rò rỉ dữ liệu từ tập test vào quá trình huấn luyện và đảm bảo dữ liệu runtime được xử lý giống dữ liệu huấn luyện.

Trong file `model.pkl`, dự án lưu cả scaler và model:

```python
joblib.dump({"scaler": scaler, "model": model}, MODEL_FILE)
```

Khi Flask chạy, `app.py` tải lại bundle này:

```python
bundle = joblib.load("model.pkl")
scaler = bundle["scaler"]
model = bundle["model"]
```

Do đó, mọi request dự đoán đều được chuẩn hóa bằng đúng scaler đã học từ tập train.

### 2.4. Logistic Regression

Logistic Regression là thuật toán học máy dùng cho bài toán phân loại. Mặc dù tên gọi có chữ "Regression", thuật toán này thường được sử dụng để dự đoán xác suất một mẫu thuộc về một lớp nào đó.

Với bài toán phân loại nhị phân, Logistic Regression học một hàm tuyến tính của các đặc trưng:

```text
score = w1*x1 + w2*x2 + ... + wn*xn + b
```

Sau đó, điểm số này được đưa qua hàm sigmoid để tạo giá trị xác suất:

```text
P(y = 1 | x) = 1 / (1 + e^(-score))
```

Trong bài toán nhận diện chữ số, mỗi ảnh sau khi flatten có `784` đặc trưng. Mô hình học trọng số tương ứng với từng pixel để xác định mẫu đầu vào có khả năng thuộc về chữ số nào.

### 2.5. Logistic Regression one-vs-rest

MNIST là bài toán phân loại đa lớp với 10 lớp chữ số. Logistic Regression cơ bản thường được trình bày cho bài toán nhị phân, vì vậy cần một chiến lược mở rộng sang đa lớp. Dự án sử dụng chiến lược one-vs-rest thông qua tham số:

```python
model = LogisticRegression(multi_class="ovr", max_iter=100, n_jobs=1)
```

One-vs-rest hoạt động như sau:

- Với mỗi chữ số, hệ thống huấn luyện một bộ phân loại nhị phân riêng.
- Ví dụ, bộ phân loại cho chữ số `3` học cách phân biệt "là 3" và "không phải 3".
- Tương tự, hệ thống có các bộ phân loại cho các lớp `0` đến `9`.
- Khi dự đoán, tất cả bộ phân loại cùng tính điểm cho mẫu đầu vào.
- Lớp có điểm dự đoán cao nhất được chọn làm kết quả cuối cùng.

Lý do chọn Logistic Regression one-vs-rest trong đề tài:

- Thuật toán đơn giản, dễ giải thích và phù hợp với mục tiêu học thuật.
- Thời gian huấn luyện tương đối nhanh trên MNIST.
- Có sẵn implementation ổn định trong scikit-learn.
- Phù hợp để minh họa toàn bộ pipeline từ tiền xử lý, huấn luyện, đánh giá đến triển khai web API.
- Kết quả đạt `0.9175` accuracy trên tập test, đủ tốt cho một hệ thống demo cơ bản.

Tuy vậy, Logistic Regression là mô hình tuyến tính nên khả năng khai thác cấu trúc không gian của ảnh còn hạn chế so với các mô hình học sâu như Convolutional Neural Network.

### 2.6. Độ đo Accuracy

Độ chính xác Accuracy được tính bằng tỷ lệ mẫu dự đoán đúng trên tổng số mẫu kiểm thử:

```text
Accuracy = số mẫu dự đoán đúng / tổng số mẫu kiểm thử
```

Trong `train.py`, accuracy được tính bằng hàm `accuracy_score`:

```python
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
```

Kết quả thực nghiệm từ `report.txt`:

- Số lượng ảnh train: `60.000`
- Số lượng ảnh test: `10.000`
- Thời gian huấn luyện: `20.97 giây`
- Accuracy trên tập test: `0.9175`

## Chương 3: Xây dựng hệ thống

### 3.1. Huấn luyện mô hình

Quá trình huấn luyện được triển khai trong file `train.py`.

Đầu tiên, chương trình tải dữ liệu MNIST. Nếu file cache `mnist_784.pkl` đã tồn tại, dữ liệu được đọc trực tiếp từ cache:

```python
mnist = joblib.load(MNIST_CACHE_FILE)
return mnist["data"], mnist["target"]
```

Nếu chưa có cache, chương trình tải dữ liệu từ OpenML:

```python
mnist = fetch_openml(
    "mnist_784",
    version=1,
    cache=True,
    as_frame=False,
    data_home=OPENML_CACHE_DIR,
    parser="liac-arff",
)
```

Sau khi tải, dữ liệu được chia thành tập train và test:

```python
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
```

Tập train gồm `60.000` ảnh đầu tiên, tập test gồm `10.000` ảnh còn lại.

Tiếp theo, chương trình lưu một số ảnh mẫu vào thư mục `images/` nhằm phục vụ minh họa dữ liệu:

- `train_sample_0_label_5.png`
- `train_sample_1_label_0.png`
- `train_sample_2_label_4.png`
- `train_sample_3_label_1.png`
- `train_sample_4_label_9.png`
- `mnist_samples_montage.png`

Sau đó, dữ liệu được chuẩn hóa bằng `StandardScaler`, mô hình được huấn luyện bằng `LogisticRegression`, và kết quả được đánh giá trên tập test.

Cuối cùng, chương trình lưu scaler và model vào file `model.pkl`, đồng thời ghi báo cáo kết quả vào `report.txt`.

### 3.2. Backend Flask

Backend được xây dựng trong file `app.py`. Khi ứng dụng khởi động, Flask tải file `model.pkl`:

```python
bundle = joblib.load("model.pkl")
scaler = bundle["scaler"]
model = bundle["model"]
```

Hàm dự đoán chính là `predict_digit`:

```python
def predict_digit(image_array):
    image_scaled = scaler.transform(image_array.reshape(1, -1))
    return int(model.predict(image_scaled)[0])
```

Hàm này thực hiện ba bước:

1. Reshape mảng ảnh thành dạng `(1, 784)` để phù hợp với scikit-learn.
2. Chuẩn hóa dữ liệu bằng scaler đã học từ tập train.
3. Gọi `model.predict` để lấy nhãn dự đoán.

Flask chạy trên host `0.0.0.0` và port `5000`:

```python
app.run(host="0.0.0.0", port=5000)
```

Thiết lập này cho phép ứng dụng có thể chạy trong Docker container và được truy cập từ bên ngoài container thông qua port mapping.

### 3.3. Giao diện Canvas

Giao diện người dùng được xây dựng trong file `static/index.html`. Trang web gồm:

- Canvas chính kích thước `280x280` để người dùng vẽ chữ số.
- Nút `Xóa` để reset Canvas.
- Nút `Dự Đoán` để gửi dữ liệu lên server.
- Canvas preview kích thước `28x28` để hiển thị dữ liệu thực tế được gửi cho mô hình.
- Khu vực hiển thị kết quả dự đoán và trạng thái xử lý.

Khi người dùng nhấn `Dự Đoán`, JavaScript thực hiện:

1. Tạo một Canvas tạm kích thước `28x28`.
2. Vẽ lại nội dung Canvas chính vào Canvas tạm.
3. Đọc dữ liệu pixel bằng `getImageData`.
4. Tính grayscale cho từng pixel.
5. Tạo mảng `784` giá trị.
6. Gửi request `POST /predict` bằng Fetch API.

Đoạn xử lý gửi request:

```javascript
const response = await fetch("/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({ image })
});
```

Kết quả trả về được hiển thị trực tiếp trên giao diện.

## Chương 4: Thiết kế API

### 4.1. Endpoint `GET /`

Endpoint này trả về file giao diện chính:

```python
@app.route("/")
def index():
    return send_from_directory("static", "index.html")
```

Người dùng truy cập `http://localhost:5000` sẽ nhận được giao diện web để vẽ chữ số.

### 4.2. Endpoint `POST /predict`

Endpoint `/predict` nhận dữ liệu ảnh đã được xử lý sẵn dưới dạng mảng `784` số.

Định nghĩa trong `app.py`:

```python
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    image = data.get("image")

    if not isinstance(image, list) or len(image) != 784:
        return jsonify({"error": "Field 'image' must be an array of 784 numbers."}), 400

    try:
        image_array = np.array(image, dtype=float)
    except (TypeError, ValueError):
        return jsonify({"error": "Field 'image' must contain only numbers."}), 400

    prediction = predict_digit(image_array)

    return jsonify({"prediction": prediction})
```

#### Request hợp lệ

```json
{
  "image": [0, 0, 0, 0, 12, 35, 255]
}
```

Trong request thực tế, mảng `image` phải có đúng `784` phần tử. Ví dụ trên chỉ rút gọn để minh họa định dạng.

#### Response thành công

```json
{
  "prediction": 7
}
```

#### Response lỗi khi thiếu hoặc sai kích thước mảng

```json
{
  "error": "Field 'image' must be an array of 784 numbers."
}
```

#### Response lỗi khi mảng chứa giá trị không phải số

```json
{
  "error": "Field 'image' must contain only numbers."
}
```

Endpoint này là endpoint chính được giao diện Canvas sử dụng.

### 4.3. Endpoint `POST /predict_base64`

Endpoint `/predict_base64` nhận ảnh dạng base64. Đây là phương án phù hợp nếu client muốn gửi trực tiếp ảnh thay vì tự chuyển ảnh thành mảng pixel.

Định nghĩa trong `app.py`:

```python
@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    data = request.get_json(silent=True) or {}

    if "image_base64" not in data:
        return jsonify({"error": "Field 'image_base64' is required."}), 400

    try:
        image = decode_base64_image(data["image_base64"])
        image_array = image_to_model_input(image)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    prediction = predict_digit(image_array)

    return jsonify({"prediction": prediction})
```

Luồng xử lý của endpoint:

1. Kiểm tra request có trường `image_base64`.
2. Giải mã chuỗi base64 thành bytes.
3. Dùng Pillow để mở ảnh.
4. Chuyển ảnh sang grayscale.
5. Resize ảnh về `28x28`.
6. Chuyển ảnh thành mảng NumPy kiểu `float`.
7. Flatten thành vector `784` chiều.
8. Chuẩn hóa bằng `StandardScaler`.
9. Dự đoán bằng Logistic Regression.
10. Trả kết quả JSON.

#### Request hợp lệ

```json
{
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

Endpoint cũng hỗ trợ chuỗi base64 không có prefix `data:image/...;base64,`.

#### Response thành công

```json
{
  "prediction": 3
}
```

#### Response lỗi khi thiếu trường `image_base64`

```json
{
  "error": "Field 'image_base64' is required."
}
```

#### Response lỗi khi base64 không hợp lệ

```json
{
  "error": "Field 'image_base64' must contain a valid base64 image."
}
```

#### Response lỗi khi giá trị rỗng

```json
{
  "error": "Field 'image_base64' must be a non-empty base64 string."
}
```

### 4.4. Kiểm tra dữ liệu đầu vào

Hệ thống có các kiểm tra cơ bản nhằm tránh lỗi runtime:

- `/predict` yêu cầu trường `image` là danh sách có đúng `784` phần tử.
- `/predict` ép kiểu mảng sang `float`; nếu thất bại, API trả lỗi `400`.
- `/predict_base64` yêu cầu trường `image_base64` tồn tại.
- `/predict_base64` kiểm tra chuỗi base64 không rỗng.
- `/predict_base64` chỉ chấp nhận ảnh base64 có thể mở được bằng Pillow.

Các lỗi đều được trả về dưới dạng JSON, giúp client xử lý thống nhất.

## Chương 5: Triển khai Docker và Cloud

### 5.1. Lý do sử dụng Docker

Docker giúp đóng gói ứng dụng cùng môi trường chạy và các thư viện phụ thuộc. Với dự án này, Docker có các lợi ích:

- Đảm bảo môi trường chạy nhất quán giữa máy phát triển và server.
- Giảm lỗi do thiếu thư viện hoặc khác phiên bản Python.
- Dễ triển khai lên các nền tảng cloud hỗ trợ container.
- Có thể chạy ứng dụng chỉ với một số lệnh build và run.

### 5.2. Nội dung Dockerfile

Dockerfile của dự án:

```dockerfile
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

Ý nghĩa các bước:

- `FROM python:3.9-slim`: sử dụng image nền Python 3.9 bản gọn nhẹ.
- `ENV PYTHONDONTWRITEBYTECODE=1`: không tạo file `.pyc`.
- `ENV PYTHONUNBUFFERED=1`: in log ra console ngay lập tức, phù hợp môi trường container.
- `WORKDIR /app`: đặt thư mục làm việc trong container.
- `COPY requirements.txt .`: copy file phụ thuộc trước để tận dụng cache Docker layer.
- `RUN pip install --no-cache-dir -r requirements.txt`: cài thư viện Python.
- `COPY . .`: copy toàn bộ mã nguồn vào container.
- `EXPOSE 5000`: khai báo port ứng dụng Flask.
- `CMD ["python", "app.py"]`: lệnh chạy ứng dụng.

### 5.3. Các thư viện phụ thuộc

File `requirements.txt` gồm:

```text
flask
scikit-learn
numpy
pillow
joblib
```

Vai trò của từng thư viện:

- `flask`: xây dựng web server và API.
- `scikit-learn`: tải MNIST, chuẩn hóa dữ liệu, huấn luyện Logistic Regression và tính accuracy.
- `numpy`: xử lý mảng số và vector pixel.
- `pillow`: xử lý ảnh base64, chuyển grayscale, resize ảnh.
- `joblib`: lưu và tải dữ liệu cache, scaler và model.

### 5.4. Build và chạy ứng dụng bằng Docker

Từ thư mục gốc của dự án, build Docker image:

```bash
docker build -t handwritten-digit-recognition .
```

Chạy container:

```bash
docker run -p 5000:5000 handwritten-digit-recognition
```

Sau khi container chạy, truy cập ứng dụng tại:

```text
http://localhost:5000
```

Khi triển khai cloud, có thể cấu hình nền tảng cloud build trực tiếp từ Dockerfile trong repository GitHub. URL cloud sau khi triển khai có thể điền vào placeholder:

```markdown
<!-- URL_DEMO -->
```

## Chương 6: Kết quả và đánh giá

### 6.1. Kết quả huấn luyện

Kết quả được trích từ file `report.txt`:

| Chỉ số | Giá trị |
|---|---:|
| Số lượng ảnh train | 60.000 |
| Số lượng ảnh test | 10.000 |
| Thời gian huấn luyện | 20.97 giây |
| Accuracy trên tập test | 0.9175 |

Nội dung báo cáo huấn luyện cho thấy mô hình Logistic Regression one-vs-rest đã được huấn luyện trên tập train và đánh giá trên tập test. Accuracy `0.9175` nghĩa là mô hình dự đoán đúng khoảng `91,75%` số mẫu trong tập kiểm thử.

### 6.2. Đánh giá kết quả

Kết quả `0.9175` là mức phù hợp đối với một mô hình tuyến tính như Logistic Regression trên bài toán MNIST. Mô hình có ưu điểm là đơn giản, dễ triển khai, dễ giải thích và huấn luyện nhanh. Thời gian huấn luyện `20.97 giây` cho thấy giải pháp phù hợp với mục tiêu minh họa quy trình xây dựng hệ thống học máy hoàn chỉnh.

Hệ thống không chỉ dừng ở việc huấn luyện mô hình mà đã triển khai được toàn bộ luồng ứng dụng:

- Tải và cache dữ liệu MNIST.
- Tiền xử lý bằng flatten và StandardScaler.
- Huấn luyện Logistic Regression one-vs-rest.
- Lưu scaler và model vào `model.pkl`.
- Xây dựng API Flask để dự đoán.
- Xây dựng giao diện Canvas để người dùng tương tác trực tiếp.
- Đóng gói bằng Docker để triển khai.

### 6.3. Hạn chế

Dù đạt kết quả tương đối tốt, hệ thống vẫn còn một số hạn chế:

- Logistic Regression là mô hình tuyến tính nên chưa khai thác tốt cấu trúc không gian hai chiều của ảnh.
- Dữ liệu người dùng vẽ trên Canvas có thể khác đáng kể so với dữ liệu MNIST về vị trí, tỷ lệ và độ dày nét.
- Giao diện hiện tại mới hiển thị nhãn dự đoán cuối cùng, chưa hiển thị xác suất của từng lớp.
- Chưa có ma trận nhầm lẫn để phân tích sâu các cặp chữ số dễ bị dự đoán sai.
- Khi triển khai production, Flask development server chưa phải lựa chọn tối ưu; nên dùng WSGI server như Gunicorn.

### 6.4. Hướng phát triển

Các hướng cải tiến có thể thực hiện:

- Thay Logistic Regression bằng Convolutional Neural Network để cải thiện accuracy.
- Bổ sung bước crop vùng có nét vẽ, căn giữa chữ số và chuẩn hóa kích thước nét.
- Hiển thị xác suất dự đoán cho cả 10 lớp chữ số.
- Bổ sung logging, unit test và API test.
- Tối ưu Docker image và cấu hình server production.
- Triển khai CI/CD để tự động build và deploy khi cập nhật mã nguồn.

## Kết luận

Đề tài đã xây dựng thành công một hệ thống nhận diện chữ số viết tay hoàn chỉnh theo mô hình client-server. Phần huấn luyện sử dụng tập dữ liệu MNIST với `60.000` ảnh train và `10.000` ảnh test. Dữ liệu ảnh được biểu diễn dưới dạng vector `784` chiều, chuẩn hóa bằng `StandardScaler`, sau đó được huấn luyện bằng Logistic Regression theo chiến lược one-vs-rest.

Kết quả thực nghiệm đạt accuracy `0.9175` trên tập test với thời gian huấn luyện `20.97 giây`. Trên tầng ứng dụng, Flask API nhận dữ liệu từ giao diện Canvas hoặc ảnh base64, tiền xử lý dữ liệu, gọi mô hình và trả về kết quả JSON. Dockerfile giúp đóng gói ứng dụng để dễ dàng chạy trên môi trường khác hoặc triển khai lên cloud.

Mặc dù mô hình hiện tại còn hạn chế so với các phương pháp học sâu, dự án đã thể hiện đầy đủ quy trình xây dựng một hệ thống học máy ứng dụng: từ dữ liệu, tiền xử lý, huấn luyện, đánh giá, xây dựng API, giao diện người dùng đến đóng gói triển khai.

## Tài liệu tham khảo

1. Scikit-learn Documentation. <https://scikit-learn.org/stable/>
2. Scikit-learn `LogisticRegression`. <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>
3. Scikit-learn `StandardScaler`. <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>
4. Scikit-learn `fetch_openml`. <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html>
5. Flask Documentation. <https://flask.palletsprojects.com/>
6. Flask API Reference. <https://flask.palletsprojects.com/en/stable/api/>
7. OpenML MNIST Dataset. <https://www.openml.org/search?type=data&sort=runs&id=554>
8. MNIST Database of Handwritten Digits. <http://yann.lecun.com/exdb/mnist/>
9. Docker Documentation. <https://docs.docker.com/>
10. Dockerfile Reference. <https://docs.docker.com/reference/dockerfile/>
11. Pillow Documentation. <https://pillow.readthedocs.io/>
12. NumPy Documentation. <https://numpy.org/doc/>
13. Joblib Documentation. <https://joblib.readthedocs.io/>
14. MDN Web Docs: Canvas API. <https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API>
15. MDN Web Docs: Fetch API. <https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API>
