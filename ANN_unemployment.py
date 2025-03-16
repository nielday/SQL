import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Bước 1: Đọc dữ liệu
file_path = 'data.csv'
df = pd.read_csv(file_path)
print("📌 Các cột trong dữ liệu:", df.columns)

# Bước 2: Kiểm tra và xử lý dữ liệu thiếu
plt.figure(figsize=(8, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Biểu đồ hiển thị giá trị bị thiếu")
plt.show()

# Xử lý dữ liệu thiếu thay vì loại bỏ hoàn toàn
for column in df.columns:
    if df[column].isnull().sum() > 0:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)  # Thay thế bằng giá trị phổ biến nhất
        else:
            df[column].fillna(df[column].mean(), inplace=True)  # Thay thế bằng trung bình

# Mã hóa biến phân loại
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Xác định cột mục tiêu
target_column = 'Attrition'

# Kiểm tra sự tồn tại của cột mục tiêu
if target_column not in df.columns:
    raise ValueError(f"❌ Cột '{target_column}' không tồn tại trong dữ liệu!")

# Bước 3 : Chia dữ liệu train/test với stratify
test_ratio = 0.2
train_data, test_data = train_test_split(df, test_size=test_ratio, random_state=42, stratify=df[target_column])

print(f"📌 Số lượng mẫu trong tập train: {len(train_data)}")
print(f"📌 Số lượng mẫu trong tập test: {len(test_data)}")

# Lưu dữ liệu đã xử lý
train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)

# Bước 4: Khởi tạo mô hình
# Chuẩn hoá dữ liệu
X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]
X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Xây dựng mô hình ANN cải tiến
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Cài đặt Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Biểu đồ đánh giá quá trình huấn luyện
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Biểu đồ Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Biểu đồ Loss')
plt.legend()
plt.show()

# Bước 5: Thử nghiệm  và đánh giá mô hình
predictions = model.predict(X_test)
predictions = [1 if p > 0.5 else 0 for p in predictions]

# Độ chính xác mô hình
accuracy = np.mean(np.array(predictions) == np.array(y_test)) * 100
print(f'🎯 Độ chính xác trên tập kiểm tra: {accuracy:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Confusion Matrix")
plt.show()

# Báo cáo chi tiết về mô hình
print("\n📊 Báo cáo phân loại:\n", classification_report(y_test, predictions))