import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import random
import numpy as np
import tensorflow as tf
import threading
import os
import sys

sys.setrecursionlimit(5000)
def get_resource_path(relative_path):
    """ Lấy đường dẫn tuyệt đối của tài nguyên, dùng được cho cả lúc chạy code lẫn file exe """
    try:
        # PyInstaller tạo ra thư mục tạm này
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
MODEL_PATH = get_resource_path("my_autoencoder.h5")
# CẤU HÌNH
PATCH_SIZE = 64
STRIDE = 32
#MODEL_PATH = "my_autoencoder.h5"
APP_TITLE = " Khử nhiễu "

# Biến toàn cục
model = None
original_clean_img = None   # Biến này luôn giữ ảnh gốc sạch
current_noisy_img = None    # Biến này chứa ảnh đang hiển thị (Input cho AI)
current_denoised_img = None
# LOAD MODEL
def load_model_ai():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Đã load model thành công!!!")
            return True
        except Exception as e:
            print(f"Lỗi load model: {e}")
            return False
    return False

# LOGIC XỬ LÝ ẢNH

def denoise_image_logic(noisy_img):
    if model is None: return noisy_img
    
    H, W, C = noisy_img.shape
    out = np.zeros_like(noisy_img, dtype=np.float32)
    weight = np.zeros((H, W, 1), dtype=np.float32)

    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            patch = noisy_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            input_patch = np.expand_dims(patch, axis=0)
            pred = model.predict(input_patch, verbose=0)[0]
            
            out[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred
            weight[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1.0

    weight[weight == 0] = 1.0
    out /= weight
    return np.clip(out, 0.0, 1.0)

def add_random_noise(img):
    import random # Import ở đây cho chắc chắn
    
    # Định nghĩa các loại nhiễu (Đã giảm intensity xuống một chút cho đỡ nát ảnh)
    def _gaussian_noise(image):
        mean = 0
        var = random.uniform(0.018, 0.01) # Giảm var
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        return image + gauss

    def _salt_noise(image):
        amount = random.uniform(0.010, 0.02) # Giảm amount
        out = np.copy(image)
        num_salt = np.ceil(amount * image.size)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 1.0
        return out

    def _pepper_noise(image):
        amount = random.uniform(0.010, 0.02) # Giảm amount
        out = np.copy(image)
        num_pepper = np.ceil(amount * image.size)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0.0
        return out

    def _speckle_noise(image):
        mean = 0
        var = random.uniform(0.01, 0.03)
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        return image + image * gauss

    available_noises = [_gaussian_noise, _salt_noise, _pepper_noise, _speckle_noise]

    # Logic chồng lớp nhiễu
    noisy_img = img.copy()
    
    # Random nhiễu
    num_layers = random.randint(2, 3)
    
    # Chọn ngẫu nhiên
    chosen_funcs = random.choices(available_noises, k=num_layers)
    print(f"--- Đang tạo {num_layers} lớp nhiễu")
    
    for func in chosen_funcs:
        print(f" + Thêm nhiễu: {func.__name__}")
        noisy_img = func(noisy_img)
        noisy_img = np.clip(noisy_img, 0.0, 1.0)

    return noisy_img.astype(np.float32)
# XỬ LÝ GIAO DIỆN
def show_image(img_arr, label_widget, max_size=(380, 380)):
    h, w = img_arr.shape[:2]
    ratio = min(max_size[0]/w, max_size[1]/h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    
    img_resized = cv2.resize(img_arr, (new_w, new_h))
    img_pil = Image.fromarray(img_resized)
    img_tk = ImageTk.PhotoImage(img_pil)
    
    label_widget.config(image=img_tk, text="")
    label_widget.image = img_tk

def select_image():
    global current_noisy_img, original_clean_img # Gọi cả 2 biến toàn cục
    
    path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.jpeg *.bmp")])
    if not path: return
    
    # Đọc ảnh bằng OpenCV
    img = cv2.imread(path)
    if img is None: return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Chuẩn hóa về 0.0 - 1.0
    img_normalized = img.astype(np.float32) / 255.0
    
    # Lưu bản gốc sạch vào biến riêng
    original_clean_img = img_normalized
    
    # Lúc đầu chưa bấm nút nhiễu thì ảnh hiện tại = ảnh gốc
    current_noisy_img = img_normalized.copy() 
    
    show_image(img, lbl_input)
    
    # Reset giao diện
    lbl_output.config(image='', text="[Chờ xử lý...]")
    lbl_status.config(text=f"📂 Đã chọn: {os.path.basename(path)}", fg="#2980b9")
    btn_run.config(state="normal", bg="#27ae60")
    btn_noise.config(state="normal", bg="#f39c12")
# Hàm xử lý Nhiễu
def trigger_add_noise():
    global current_noisy_img, original_clean_img
    
    # Kiểm tra xem đã có ảnh gốc chưa
    if original_clean_img is None: 
        messagebox.showwarning("Chưa có ảnh", "Vui lòng chọn ảnh gốc trước!")
        return
    # Gọi hàm random nhiễu 
    noisy_result = add_random_noise(original_clean_img)
    
    # Cập nhật ảnh hiện tại thành ảnh vừa random xong
    current_noisy_img = noisy_result
    
    # Hiển thị lên màn hình
    img_display = (current_noisy_img * 255).astype(np.uint8)
    show_image(img_display, lbl_input)
    
    lbl_status.config(text="⚡ Đã tạo nhiễu ngẫu nhiên mới!", fg="#d35400")
def processing_thread():
    global current_denoised_img
    btn_run.config(state="disabled", bg="#95a5a6")
    btn_select.config(state="disabled")
    btn_noise.config(state="disabled") # Khóa nút nhiễu khi đang chạy
    
    lbl_status.config(text=" Đang khử nhiễu... Vui lòng đợi...", fg="#e67e22")
    progress.start(15)
    
    try:
        current_denoised_img = denoise_image_logic(current_noisy_img)
        
        img_uint8 = (current_denoised_img * 255).astype(np.uint8)
        show_image(img_uint8, lbl_output)
        
        lbl_status.config(text=" Phục chế hoàn tất!", fg="#27ae60")
        messagebox.showinfo("Thông báo", "Đã xử lý xong!")
        
    except Exception as e:
        print(e)
        lbl_status.config(text=" Có lỗi xảy ra!", fg="red")
        
    progress.stop()
    btn_run.config(state="normal", bg="#27ae60")
    btn_select.config(state="normal")
    btn_noise.config(state="normal")

def run_denoise():
    if current_noisy_img is None: return
    if model is None:
        messagebox.showerror("Lỗi", "Chưa tìm thấy file model .h5!")
        return
    threading.Thread(target=processing_thread).start()

def save_image():
    if current_denoised_img is None: return
    path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
    if path:
        img_save = (current_denoised_img * 255).astype(np.uint8)
        img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_save)
        messagebox.showinfo("Lưu ảnh", "Đã lưu thành công!")

# THIẾT KẾ GIAO DIỆN CHÍNH
root = tk.Tk()
root.title(APP_TITLE)
root.geometry("1000x700")
root.configure(bg="#ecf0f1")

# Header
header_frame = tk.Frame(root, bg="#2c3e50", pady=15)
header_frame.pack(fill="x")
tk.Label(header_frame, text="HỆ THỐNG KHỬ NHIỄU ẢNH", font=("Segoe UI", 18, "bold"), fg="white", bg="#2c3e50").pack()
tk.Label(header_frame, text="Version: 0.1.2 ", font=("Segoe UI", 10), fg="#bdc3c7", bg="#2c3e50").pack()

# Khu vực hiển thị ảnh
main_frame = tk.Frame(root, bg="#ecf0f1")
main_frame.pack(expand=True, fill="both", padx=20, pady=10)

# Khung trái (Input)
frame_left = tk.LabelFrame(main_frame, text=" Input (Ảnh Gốc / Nhiễu) ", font=("Arial", 11, "bold"), bg="white", fg="#333")
frame_left.pack(side="left", expand=True, fill="both", padx=10, pady=5)
lbl_input = tk.Label(frame_left, text="Chưa chọn ảnh", bg="#ecf0f1", fg="#7f8c8d", font=("Arial", 12))
lbl_input.pack(expand=True, fill="both", padx=5, pady=5)

# Khung phải (Output)
frame_right = tk.LabelFrame(main_frame, text=" Output (Đã khử nhiễu) ", font=("Arial", 11, "bold"), bg="white", fg="#27ae60")
frame_right.pack(side="right", expand=True, fill="both", padx=10, pady=5)
lbl_output = tk.Label(frame_right, text="Waiting...", bg="#ecf0f1", fg="#7f8c8d", font=("Arial", 12))
lbl_output.pack(expand=True, fill="both", padx=5, pady=5)

# Khu vực điều khiển
control_frame = tk.Frame(root, bg="#ecf0f1", pady=10)
control_frame.pack(fill="x")

style = ttk.Style()
style.theme_use('clam')
style.configure("green.Horizontal.TProgressbar", foreground='#27ae60', background='#27ae60')
progress = ttk.Progressbar(control_frame, orient="horizontal", length=800, mode="indeterminate", style="green.Horizontal.TProgressbar")
progress.pack(pady=5)

btn_frame = tk.Frame(control_frame, bg="#ecf0f1")
btn_frame.pack(pady=10)

btn_style = {"font": ("Segoe UI", 11), "width": 16, "pady": 5}

# Nút Chọn Ảnh
btn_select = tk.Button(btn_frame, text=" Chọn Ảnh", command=select_image, bg="white", **btn_style)
btn_select.pack(side="left", padx=10)

# Nút Tạo Nhiễu
btn_noise = tk.Button(btn_frame, text=" Thêm Nhiễu", command=trigger_add_noise, bg="#f39c12", fg="white", state="disabled", **btn_style)
btn_noise.pack(side="left", padx=10)

# Nút Run
btn_run = tk.Button(btn_frame, text=" Bắt Đầu Xử Lý", command=run_denoise, bg="#27ae60", fg="white", state="disabled", **btn_style)
btn_run.pack(side="left", padx=10)

# Nút Lưu
btn_save = tk.Button(btn_frame, text=" Lưu Kết Quả", command=save_image, bg="white", **btn_style)
btn_save.pack(side="left", padx=10)

# Status Bar
status_frame = tk.Frame(root, bg="#dfe6e9", height=25)
status_frame.pack(side="bottom", fill="x")
lbl_status = tk.Label(status_frame, text="Sẵn sàng", bg="#dfe6e9", fg="#2d3436", font=("Segoe UI", 9), anchor="w", padx=10)
lbl_status.pack(fill="both")

threading.Thread(target=load_model_ai).start()

root.mainloop()