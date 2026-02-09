#Vẽ mặt đồng hồ hình tròn, nền màu tím, có các số dạng la mã màu sắc khác nhau. 
#Có 3 kim đồng hồ: Giờ, phút, giây.
#Kim giờ màu xanh dương, kim phút màu xanh lá cây, kim giây màu đỏ.
#level 2: Vẽ kim giây chuyển động.
#level 3: Vẽ kim phút chuyển động.
#level 4: Vẽ kim giờ chuyển động.   
#level 5: Vẽ thêm các vạch chỉ phút trên mặt đồng hồ. Và kim giây, kim giờ, kim phút, hoạt động theo logic
import cv2 as cv
import numpy as np
import math
from datetime import datetime

# Cài đặt kích thước
WIDTH = 800
HEIGHT = 800
CENTER = (WIDTH // 2, HEIGHT // 2)
RADIUS = 300

# Màu sắc (BGR format)
PURPLE = (128, 0, 128)
BLUE = (255, 0, 0)      # Xanh dương (BGR)
GREEN = (0, 255, 0)     # Xanh lá cây
RED = (0, 0, 255)       # Đỏ
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GOLD = (0, 215, 255)    # Vàng

# La mã số
ROMAN_NUMERALS = ['XII', 'I', 'II', 'III', 'IV', 'V', 'VI', 
                   'VII', 'VIII', 'IX', 'X', 'XI']

def draw_clock_face(img):
    """Vẽ mặt đồng hồ"""
    # Vẽ hình tròn nền tím
    cv.circle(img, CENTER, RADIUS, PURPLE, -1)
    cv.circle(img, CENTER, RADIUS, WHITE, 3)
    
    # Vẽ các số la mã
    for i in range(12):
        angle = math.radians(i * 30 - 90)  # -90 để bắt đầu từ 12
        x = int(CENTER[0] + (RADIUS - 60) * math.cos(angle))
        y = int(CENTER[1] + (RADIUS - 60) * math.sin(angle))
        
        # Màu sắc khác nhau cho mỗi số
        if i % 3 == 0:
            color = GOLD
        elif i % 3 == 1:
            color = WHITE
        else:
            color = (200, 200, 200)
        
        cv.putText(img, ROMAN_NUMERALS[i], (x - 15, y + 15), 
                  cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

def draw_minute_marks(img):
    """Vẽ các vạch chỉ phút"""
    for i in range(60):
        angle = math.radians(i * 6 - 90)  # 6 độ cho mỗi phút
        
        # Vạch dài cho giờ (mỗi 5 phút)
        if i % 5 == 0:
            start_radius = RADIUS - 40
            end_radius = RADIUS - 15
            thickness = 3
            color = WHITE
        else:
            start_radius = RADIUS - 30
            end_radius = RADIUS - 20
            thickness = 1
            color = (200, 200, 200)
        
        x1 = int(CENTER[0] + start_radius * math.cos(angle))
        y1 = int(CENTER[1] + start_radius * math.sin(angle))
        x2 = int(CENTER[0] + end_radius * math.cos(angle))
        y2 = int(CENTER[1] + end_radius * math.sin(angle))
        
        cv.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_hand(img, angle, length, color, thickness):
    """Vẽ kim đồng hồ
    angle: góc (độ) từ 12 giờ (0 độ)
    length: độ dài kim
    """
    angle_rad = math.radians(angle - 90)
    end_x = int(CENTER[0] + length * math.cos(angle_rad))
    end_y = int(CENTER[1] + length * math.sin(angle_rad))
    cv.line(img, CENTER, (end_x, end_y), color, thickness)

def draw_center_circle(img):
    """Vẽ hình tròn ở tâm"""
    cv.circle(img, CENTER, 10, BLACK, -1)
    cv.circle(img, CENTER, 10, WHITE, 2)

def main():
    while True:
        # Tạo ảnh mới
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # Vẽ mặt đồng hồ
        draw_clock_face(img)
        draw_minute_marks(img)
        
        # Lấy thời gian hiện tại
        now = datetime.now()
        hours = now.hour % 12
        minutes = now.minute
        seconds = now.second
        milliseconds = now.microsecond // 1000
        
        # Tính toán góc cho mỗi kim
        # Kim giây: 360 độ / 60 giây = 6 độ/giây
        second_angle = (seconds + milliseconds / 1000) * 6
        
        # Kim phút: 360 độ / 60 phút = 6 độ/phút
        minute_angle = (minutes + seconds / 60) * 6
        
        # Kim giờ: 360 độ / 12 giờ = 30 độ/giờ
        hour_angle = (hours + minutes / 60) * 30
        
        # Vẽ kim
        draw_hand(img, hour_angle, 150, BLUE, 6)      # Kim giờ - xanh dương
        draw_hand(img, minute_angle, 200, GREEN, 4)   # Kim phút - xanh lá cây
        draw_hand(img, second_angle, 220, RED, 2)     # Kim giây - đỏ
        
        # Vẽ hình tròn ở tâm
        draw_center_circle(img)
        
        # Hiển thị thời gian hiện tại
        time_str = now.strftime("%H:%M:%S")
        cv.putText(img, time_str, (WIDTH - 200, 50), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
        
        # Hiển thị ảnh
        cv.imshow('Analog Clock', img)
        
        # Nhấn ESC để thoát
        if cv.waitKey(1) == 27:
            break
    
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
    