"""
Chương trình đếm số lượng hình tròn vượt qua line màu đỏ trên băng chuyền
Sử dụng OpenCV với Background Subtraction và Object Tracking
"""

import cv2 as cv
import numpy as np


def main():
    # ==================== KHỞI TẠO VIDEO ====================
    # Đọc video từ file
    vid = cv.VideoCapture("bang_chuyen.mp4")
    
    # Lấy thông tin video
    frame_width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv.CAP_PROP_FPS))
    
    print(f"Video: {frame_width}x{frame_height}, FPS: {fps}")
    
    # ==================== THIẾT LẬP THAM SỐ ====================
    # Vị trí line đếm (65% chiều rộng từ trái sang phải)
    line_position = int(frame_width * 0.65)
    
    # Biến đếm số vật thể đã qua line
    count = 0
    
    # Danh sách theo dõi các object
    # Mỗi object có: id, danh sách vị trí (positions), và trạng thái đã đếm (counted)
    tracked_objects = []
    next_object_id = 0
    
    # Khởi tạo Background Subtractor (phát hiện chuyển động)
    bg_subtractor = cv.createBackgroundSubtractorMOG2(
        history=100, 
        varThreshold=40, 
        detectShadows=False
    )
    
    frame_count = 0
    
    print(f"Line đếm ở vị trí: x = {line_position}")
    print("Nhấn 'q' để thoát\n")
    
    # ==================== XỬ LÝ VIDEO ====================
    while True:
        # Đọc frame
        ret, frame = vid.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Tạo bản sao để vẽ kết quả
        display_frame = frame.copy()
        
        # --- BƯỚC 1: PHÁT HIỆN VẬT THỂ ---
        # Áp dụng background subtraction để tách foreground
        fg_mask = bg_subtractor.apply(frame)
        
        # Threshold để loại bỏ noise và chỉ giữ lại vật thể rõ ràng
        _, fg_mask = cv.threshold(fg_mask, 250, 255, cv.THRESH_BINARY)
        
        # Morphological operations để làm mịn mask
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)
        
        # Tìm contours (đường viền) của các vật thể
        contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Vẽ line màu đỏ để đếm
        cv.line(display_frame, (line_position, 0), (line_position, frame_height), (0, 0, 255), 3)
        
        # Danh sách tâm các vật thể phát hiện được trong frame này
        detected_centers = []
        
        # --- BƯỚC 2: LỌC VÀ XÁC ĐỊNH HÌNH TRÒN ---
        for contour in contours:
            # Tính diện tích
            area = cv.contourArea(contour)
            
            # Lọc theo diện tích (loại bỏ vật quá nhỏ hoặc quá lớn)
            if 800 < area < 50000:
                # Tính độ tròn (circularity) để chỉ lấy hình tròn
                perimeter = cv.arcLength(contour, True)
                if perimeter > 0:
                    # Circularity = 1 là hình tròn hoàn hảo
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    # Chỉ lấy các object đủ tròn (> 0.4)
                    if circularity > 0.4:
                        # Tính tọa độ tâm
                        M = cv.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            detected_centers.append((cx, cy))
                            
                            # Vẽ contour màu xanh lá và tâm màu xanh dương
                            cv.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
                            cv.circle(display_frame, (cx, cy), 5, (255, 0, 0), -1)
        
        # --- BƯỚC 3: TRACKING VÀ ĐẾM ---
        # Cập nhật tracking: khớp các object hiện tại với object đã track
        active_objects = []
        
        for obj in tracked_objects:
            if len(detected_centers) > 0:
                # Lấy vị trí cuối cùng của object
                last_pos = obj['positions'][-1]
                
                # Tính khoảng cách đến tất cả các center phát hiện được
                distances = [
                    np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2) 
                    for cx, cy in detected_centers
                ]
                min_dist = min(distances)
                
                # Nếu tìm thấy center gần (< 80 pixels)
                if min_dist < 80:
                    min_idx = distances.index(min_dist)
                    new_center = detected_centers[min_idx]
                    
                    # Cập nhật vị trí mới
                    obj['positions'].append(new_center)
                    
                    # Giữ tối đa 10 vị trí gần nhất (tránh list quá dài)
                    if len(obj['positions']) > 10:
                        obj['positions'] = obj['positions'][-10:]
                    
                    # Kiểm tra xem object có vượt qua line không
                    # (từ bên trái sang bên phải line)
                    if not obj['counted']:
                        if last_pos[0] < line_position <= new_center[0]:
                            count += 1
                            obj['counted'] = True
                            print(f"Frame {frame_count}: Object #{obj['id']} vượt qua line! → Tổng: {count}")
                    
                    # Giữ object này trong danh sách tracking
                    active_objects.append(obj)
                    
                    # Xóa center đã match khỏi danh sách
                    detected_centers.pop(min_idx)
        
        # Cập nhật danh sách tracking (chỉ giữ object còn active)
        tracked_objects = active_objects
        
        # Thêm object mới từ các center chưa được match
        for cx, cy in detected_centers:
            # Chỉ thêm object ở bên trái line (chưa qua line)
            if cx < line_position - 50:
                tracked_objects.append({
                    'id': next_object_id,
                    'positions': [(cx, cy)],
                    'counted': False
                })
                next_object_id += 1
        
        # --- BƯỚC 4: HIỂN THỊ KẾT QUẢ ---
        # Hiển thị số đếm
        cv.putText(display_frame, f"Count: {count}", (20, 60), 
                   cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        
        # Hiển thị số frame
        cv.putText(display_frame, f"Frame: {frame_count}", (20, 120), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Hiển thị số object đang tracking
        cv.putText(display_frame, f"Tracking: {len(tracked_objects)}", (20, 160), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # CHỈ HIỂN THỊ 1 CỬA SỔ - VIDEO VỚI LINE ĐỎ
        cv.imshow("Object Counting", display_frame)
        
        # In thông tin mỗi 50 frames
        if frame_count % 50 == 0:
            print(f"Frame: {frame_count} | Count: {count} | Tracking: {len(tracked_objects)}")
        
        # Nhấn 'q' để thoát
        if cv.waitKey(1) == ord("q"):
            break
    
    # ==================== KẾT THÚC ====================
    print(f"\n{'='*60}")
    print(f"✓ Hoàn thành!")
    print(f"{'='*60}")
    print(f"Tổng số vật thể đã qua line: {count}")
    print(f"Tổng số frames: {frame_count}")
    
    # Giải phóng tài nguyên
    vid.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()