"""
Chương trình đếm số lượng hình tròn vượt qua vùng ảo trên băng chuyền (KHÔNG VẼ LINE)
Phiên bản tối ưu với Kalman Filter, HoughCircles và thuật toán tracking nâng cao
"""

import cv2 as cv
import numpy as np
from collections import deque


class KalmanTracker:
    """Kalman Filter để tracking và dự đoán vị trí object"""
    def __init__(self, initial_position):
        self.kf = cv.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        self.kf.statePost = np.array([[initial_position[0]], 
                                       [initial_position[1]], 
                                       [0], [0]], np.float32)
        self.predicted = initial_position
        
    def predict(self):
        """Dự đoán vị trí tiếp theo"""
        prediction = self.kf.predict()
        self.predicted = (int(prediction[0][0]), int(prediction[1][0]))
        return self.predicted
    
    def update(self, measurement):
        """Cập nhật với vị trí đo được thực tế"""
        self.kf.correct(np.array([[np.float32(measurement[0])],
                                   [np.float32(measurement[1])]]))
        return measurement


class TrackedObject:
    """Đại diện cho một object đang được tracking"""
    def __init__(self, obj_id, position):
        self.id = obj_id
        self.kalman = KalmanTracker(position)
        self.positions = deque([position], maxlen=15)
        self.missed_frames = 0
        self.counted = False
        self.age = 0
        
    def predict(self):
        """Dự đoán vị trí tiếp theo"""
        return self.kalman.predict()
    
    def update(self, position):
        """Cập nhật với vị trí thực tế"""
        self.kalman.update(position)
        self.positions.append(position)
        self.missed_frames = 0
        self.age += 1
        
    def mark_missed(self):
        """Đánh dấu frame bị miss"""
        self.missed_frames += 1
        predicted = self.predict()
        self.positions.append(predicted)
        
    def get_current_position(self):
        """Lấy vị trí hiện tại"""
        return self.positions[-1]
    
    def is_lost(self, max_missed=10):
        """Kiểm tra object có bị mất không"""
        return self.missed_frames > max_missed


def detect_circles_hybrid(frame, fg_mask):
    """
    Phát hiện hình tròn bằng phương pháp kết hợp:
    1. Contour-based detection (từ background subtraction)
    2. Hough Circles (cho độ chính xác cao)
    """
    circles = []
    
    # PHƯƠNG PHÁP 1: Contour-based (nhanh, ổn định)
    contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv.contourArea(contour)
        
        if 800 < area < 50000:
            perimeter = cv.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                
                if circularity > 0.35:  # Ngưỡng tròn linh hoạt hơn
                    M = cv.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Tính bán kính ước lượng
                        radius = int(np.sqrt(area / np.pi))
                        
                        circles.append({
                            'center': (cx, cy),
                            'radius': radius,
                            'area': area,
                            'circularity': circularity,
                            'method': 'contour'
                        })
    
    # PHƯƠNG PHÁP 2: Hough Circles (chính xác cho hình tròn rõ)
    # Chỉ chạy khi cần thiết để tiết kiệm tài nguyên
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_blurred = cv.GaussianBlur(gray, (9, 9), 2)
    
    detected_circles = cv.HoughCircles(
        gray_blurred,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=15,
        maxRadius=80
    )
    
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for circle in detected_circles[0, :]:
            cx, cy, r = circle
            area = np.pi * r * r
            
            if 800 < area < 50000:
                circles.append({
                    'center': (int(cx), int(cy)),
                    'radius': int(r),
                    'area': area,
                    'circularity': 1.0,  # Hough circles luôn tròn
                    'method': 'hough'
                })
    
    # Loại bỏ trùng lặp (merge circles gần nhau)
    circles = merge_nearby_circles(circles, distance_threshold=40)
    
    return circles


def merge_nearby_circles(circles, distance_threshold=40):
    """Gộp các circles gần nhau thành một"""
    if len(circles) <= 1:
        return circles
    
    merged = []
    used = set()
    
    for i, circle1 in enumerate(circles):
        if i in used:
            continue
            
        group = [circle1]
        cx1, cy1 = circle1['center']
        
        for j, circle2 in enumerate(circles[i+1:], start=i+1):
            if j in used:
                continue
                
            cx2, cy2 = circle2['center']
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            if distance < distance_threshold:
                group.append(circle2)
                used.add(j)
        
        # Lấy circle tốt nhất trong nhóm (ưu tiên circularity cao)
        best_circle = max(group, key=lambda c: c['circularity'])
        merged.append(best_circle)
        used.add(i)
    
    return merged


def match_objects_to_detections(tracked_objects, detected_circles, max_distance=100):
    """
    Khớp objects đang track với detections mới bằng Hungarian Algorithm (simplified)
    """
    if not tracked_objects or not detected_circles:
        return [], detected_circles
    
    # Ma trận khoảng cách
    distance_matrix = np.zeros((len(tracked_objects), len(detected_circles)))
    
    for i, obj in enumerate(tracked_objects):
        predicted_pos = obj.predict()
        
        for j, circle in enumerate(detected_circles):
            cx, cy = circle['center']
            dist = np.sqrt((predicted_pos[0] - cx)**2 + (predicted_pos[1] - cy)**2)
            distance_matrix[i, j] = dist
    
    # Simple greedy matching (có thể thay bằng Hungarian nếu cần)
    matches = []
    unmatched_detections = list(range(len(detected_circles)))
    
    for obj_idx in range(len(tracked_objects)):
        if len(unmatched_detections) == 0:
            break
            
        # Tìm detection gần nhất với object này
        min_dist = float('inf')
        min_det_idx = -1
        
        for det_idx in unmatched_detections:
            if distance_matrix[obj_idx, det_idx] < min_dist:
                min_dist = distance_matrix[obj_idx, det_idx]
                min_det_idx = det_idx
        
        # Chỉ match nếu đủ gần
        if min_dist < max_distance:
            matches.append((obj_idx, min_det_idx))
            unmatched_detections.remove(min_det_idx)
    
    # Các detections chưa match
    unmatched_circles = [detected_circles[i] for i in unmatched_detections]
    
    return matches, unmatched_circles


def main():
    # ==================== KHỞI TẠO VIDEO ====================
    vid = cv.VideoCapture("bang_chuyen.mp4")
    
    if not vid.isOpened():
        print("❌ Lỗi: Không thể mở file video 'bang_chuyen.mp4'")
        return
    
    frame_width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv.CAP_PROP_FPS))
    total_frames = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {frame_width}x{frame_height}, FPS: {fps}, Tổng frames: {total_frames}")
    
    # ==================== THIẾT LẬP THAM SỐ ====================
    # Vị trí line ảo (KHÔNG VẼ trên video)
    line_position = int(frame_width * 0.65)
    
    # Biến đếm
    count = 0
    tracked_objects = []
    next_object_id = 0
    
    # Background Subtractor với tham số tối ưu
    bg_subtractor = cv.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=25,
        detectShadows=False
    )
    
    frame_count = 0
    
    print(f"⚙️  Line đếm (MÀU ĐỎ) ở vị trí: x = {line_position}")
    print("⌨️  Nhấn 'q' để thoát, 'p' để pause\n")
    print(f"{'='*70}")
    
    paused = False
    
    # ==================== XỬ LÝ VIDEO ====================
    while True:
        if not paused:
            ret, frame = vid.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Áp dụng background subtraction
            fg_mask = bg_subtractor.apply(frame, learningRate=0.01)
            
            # Xử lý mask
            _, fg_mask = cv.threshold(fg_mask, 250, 255, cv.THRESH_BINARY)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel, iterations=2)
            fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)
            
            # Phát hiện circles
            detected_circles = detect_circles_hybrid(frame, fg_mask)
            
            # Match với tracked objects
            matches, unmatched_circles = match_objects_to_detections(
                tracked_objects, detected_circles, max_distance=80
            )
            
            # Cập nhật matched objects
            updated_objects = []
            for obj_idx, det_idx in matches:
                obj = tracked_objects[obj_idx]
                circle = detected_circles[det_idx]
                
                old_pos = obj.get_current_position()
                new_pos = circle['center']
                
                obj.update(new_pos)
                
                # Kiểm tra vượt line (từ trái sang phải)
                if not obj.counted and obj.age > 3:  # Chỉ đếm sau khi track ổn định
                    if old_pos[0] < line_position <= new_pos[0]:
                        count += 1
                        obj.counted = True
                        print(f"✓ Frame {frame_count:4d}: Object #{obj.id:3d} vượt qua line → Tổng: {count}")
                
                updated_objects.append(obj)
            
            # Xử lý unmatched objects (bị miss)
            for i, obj in enumerate(tracked_objects):
                if not any(i == match[0] for match in matches):
                    obj.mark_missed()
                    if not obj.is_lost(max_missed=15):
                        updated_objects.append(obj)
            
            tracked_objects = updated_objects
            
            # Tạo objects mới từ unmatched detections
            for circle in unmatched_circles:
                cx, cy = circle['center']
                
                # Chỉ tạo object mới ở phía bên trái line
                if cx < line_position - 30:
                    new_obj = TrackedObject(next_object_id, (cx, cy))
                    tracked_objects.append(new_obj)
                    next_object_id += 1
            
            # ==================== HIỂN THỊ VỚI LINE MÀU ĐỎ ====================
            display_frame = frame.copy()
            
            # VẼ LINE MÀU ĐỎ ĐỂ ĐẾM
            cv.line(display_frame, (line_position, 0), (line_position, frame_height), (0, 0, 255), 3)
            
            # Vẽ tracking info
            for obj in tracked_objects:
                pos = obj.get_current_position()
                color = (0, 255, 0) if not obj.counted else (0, 165, 255)
                cv.circle(display_frame, pos, 8, color, -1)
                cv.circle(display_frame, pos, 20, color, 2)
            
            # Hiển thị thông tin
            cv.putText(display_frame, f"Count: {count}", (20, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (20, 100), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv.putText(display_frame, f"Tracking: {len(tracked_objects)}", (20, 135), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Progress bar
            progress = int((frame_count / total_frames) * 100)
            cv.rectangle(display_frame, (20, frame_height - 40), 
                        (20 + int(progress * 6), frame_height - 20), (0, 255, 0), -1)
            cv.putText(display_frame, f"{progress}%", (20 + int(progress * 6) + 10, frame_height - 23), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Hiển thị frame
        cv.imshow("Object Counting - Red Line Detection", display_frame)
        
        # Xử lý phím
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print("⏸️  PAUSED" if paused else "▶️  RESUMED")
        
        # Log mỗi 100 frames
        if not paused and frame_count % 100 == 0:
            print(f"📊 Frame: {frame_count}/{total_frames} | Count: {count} | Tracking: {len(tracked_objects)}")
    
    # ==================== KẾT THÚC ====================
    print(f"\n{'='*70}")
    print(f"✅ HOÀN THÀNH!")
    print(f"{'='*70}")
    print(f"🎯 Tổng số vật thể đã qua line: {count}")
    print(f"📹 Tổng số frames đã xử lý: {frame_count}")
    print(f"🎬 Tỷ lệ hoàn thành: {(frame_count/total_frames)*100:.1f}%")
    print(f"{'='*70}")
    
    vid.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()