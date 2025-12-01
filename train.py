import os
import numpy as np
import cv2
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
import random

# CẤU HÌNH
CLEAN_DIR = "data_clean"  # Thư mục chứa ảnh MỚI cần thêm vào
DATA_FILE = "training_data.npz" # Tên file lưu trữ dữ liệu đã xử lý
IMG_SIZE = 128            
PATCH_SIZE = 64
STRIDE = 32
EPOCHS = 15            
BATCH_SIZE = 16
MODEL_NAME = "my_autoencoder.h5"
RESET_DATA = False # Đặt True nếu muốn xóa dữ liệu cũ và làm lại từ đầu

# HÀM TẠO NHIỄU 

def add_noise(img, min_noises=2, max_noises=4):
    """
    Áp dụng ngẫu nhiên 4 loại nhiễu cơ bản chồng lên nhau.
    Input: img (float32, 0-1)
    Output: img (float32, 0-1)
    """
    
    # Định nghĩa 4 loại nhiễu cơ bản
    
    def _gaussian_noise(image):
        # Nhiễu Gaussian (Nhiễu nền phổ biến)
        mean = 0
        var = random.uniform(0.005, 0.02)
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        return image + gauss

    def _salt_noise(image):
        # Nhiễu Muối (Chấm trắng ngẫu nhiên)
        amount = random.uniform(0.01, 0.05)
        out = np.copy(image)
        num_salt = np.ceil(amount * image.size)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 1.0
        return out

    def _pepper_noise(image):
        # Nhiễu Tiêu (Chấm đen ngẫu nhiên)
        amount = random.uniform(0.01, 0.05)
        out = np.copy(image)
        num_pepper = np.ceil(amount * image.size)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0.0
        return out

    def _speckle_noise(image):
        # Nhiễu Speckle (Nhiễu lốm đốm - phụ thuộc vào giá trị pixel)
        # Công thức: I = I + I * Gaussian
        mean = 0
        var = random.uniform(0.01, 0.05)
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        return image + image * gauss

    # Danh sách giới hạn đúng 4 loại
    available_noises = [
        _gaussian_noise, 
        _salt_noise, 
        _pepper_noise, 
        _speckle_noise
    ]

    # Logic chọn và chồng nhiễu
    noisy_img = img.copy()
    
    # Chọn ngẫu nhiên số lượng lớp nhiễu (ví dụ: chồng 2 đến 4 lớp)
    num_layers = random.randint(min_noises, max_noises)
    
    # Lấy ngẫu nhiên các hàm nhiễu (có thể lặp lại, ví dụ: 2 lần Gaussian + 1 lần Salt)
    chosen_funcs = random.choices(available_noises, k=num_layers)
    
    for func in chosen_funcs:
        noisy_img = func(noisy_img)
        # Clip ngay sau mỗi bước để đảm bảo giá trị luôn hợp lệ (0-1)
        noisy_img = np.clip(noisy_img, 0.0, 1.0)

    return noisy_img.astype(np.float32)
# HÀM CẮT ẢNH
def extract_patches(img):
    H, W, C = img.shape
    patches = []
    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]
            patches.append(patch)
    return patches

# QUẢN LÝ DỮ LIỆU
X_final = np.array([], dtype=np.float32).reshape(0, PATCH_SIZE, PATCH_SIZE, 3)
y_final = np.array([], dtype=np.float32).reshape(0, PATCH_SIZE, PATCH_SIZE, 3)

#  Load dữ liệu cũ nếu có
if not RESET_DATA and os.path.exists(DATA_FILE):
    print(f" Đã tìm thấy file dữ liệu : {DATA_FILE}")
    try:
        data = np.load(DATA_FILE)
        X_final = data['X']
        y_final = data['y']
        print(f" -> Đã load {len(X_final)} mẫu từ lần chạy trước.")
    except Exception as e:
        print(f" -> Lỗi khi đọc file cũ: {e}. Sẽ tạo mới.")
else:
    if RESET_DATA:
        print(" [INFO] Chế độ RESET_DATA bật. Sẽ bỏ qua dữ liệu cũ.")
    else:
        print(" Chưa có dữ liệu cũ. Bắt đầu mới.")

# Xử lý ảnh mới trong thư mục CLEAN_DIR
print(f" Đang quét thư mục '{CLEAN_DIR}' để thêm dữ liệu mới...")
if os.path.exists(CLEAN_DIR):
    files = os.listdir(CLEAN_DIR)
    new_clean_patches = []
    new_noisy_patches = []
    
    count_processed = 0
    for f in files:
        try:
            path = os.path.join(CLEAN_DIR, f)
            img = cv2.imread(path)
            if img is None: continue
            
            # Resize và chuẩn hóa
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            # Tạo ảnh nhiễu
            img_noisy = add_noise(img)
            
            # Cắt nhỏ
            c_p = extract_patches(img)
            n_p = extract_patches(img_noisy)
            
            new_clean_patches.extend(c_p)
            new_noisy_patches.extend(n_p)
            count_processed += 1
        except Exception as e:
            print(f"Lỗi file {f}: {e}")

    if len(new_clean_patches) > 0:
        print(f" -> Đã xử lý {count_processed} ảnh mới, tạo ra {len(new_clean_patches)} patches.")
        X_new = np.array(new_noisy_patches)
        y_new = np.array(new_clean_patches)

        # Gộp dữ liệu (Concatenate)
        print(" Đang gộp dữ liệu mới vào dữ liệu cũ...")
        X_final = np.concatenate((X_final, X_new), axis=0) if len(X_final) > 0 else X_new
        y_final = np.concatenate((y_final, y_new), axis=0) if len(y_final) > 0 else y_new
        
        # Lưu lại ngay lập tức
        print(f" Đang lưu toàn bộ {len(X_final)} mẫu vào {DATA_FILE}...")
        np.savez_compressed(DATA_FILE, X=X_final, y=y_final)
        print(" -> Lưu thành công!")
    else:
        print(" Không có ảnh mới hoặc thư mục rỗng. Sử dụng dữ liệu cũ (nếu có).")
else:
    print(f" Không tìm thấy thư mục {CLEAN_DIR}.")

# Kiểm tra tổng dữ liệu trước khi train
if len(X_final) == 0:
    print(" LỖI: Không có dữ liệu nào để train (Không có file cũ lẫn ảnh mới).")
    exit()

print(f" TỔNG DỮ LIỆU HIỆN TẠI: {len(X_final)} patches.")

# Chia tập train/val
X_train, X_val, y_train, y_val = train_test_split(X_final, y_final, test_size=0.1, random_state=42)

# XÂY DỰNG MODEL
# Kiểm tra nếu có model cũ thì load để train tiếp (transfer learning), nếu không thì tạo mới
if os.path.exists(MODEL_NAME):
    print(f" Tìm thấy model cũ '{MODEL_NAME}', đang load để train tiếp...")
    autoencoder = models.load_model(MODEL_NAME)
else:
    print(" Tạo model mới...")
    input_img = layers.Input(shape=(PATCH_SIZE, PATCH_SIZE, 3))
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    
print(" Đang biên dịch model (Eager Execution Mode)...")
autoencoder.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)
autoencoder.summary()

# TRAIN MODEL
print(f" Bắt đầu train trong {EPOCHS} epochs...")
autoencoder.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_val, y_val)
)

# LƯU MODEL
autoencoder.save(MODEL_NAME)
print(f" Đã lưu mô hình thành công: {MODEL_NAME}")