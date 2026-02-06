# Hệ thống phát hiện chuyển động với Camera
import cv2
import numpy as np
import time
import datetime
import os
from collections import deque

class MotionDetector:
    def __init__(self, min_area=500, delta_threshold=25, blur_size=(21, 21)):
        """Khởi tạo bộ phát hiện chuyển động"""
        self.min_area = min_area
        self.delta_threshold = delta_threshold
        self.blur_size = blur_size
        self.background = None
        self.motion_counter = 0
        self.total_motion_detected = 0
        
        # Tạo thư mục lưu trữ
        self.output_dir = "motion_detection_output"
        self.video_dir = os.path.join(self.output_dir, "videos")
        self.image_dir = os.path.join(self.output_dir, "snapshots")
        
        for directory in [self.output_dir, self.video_dir, self.image_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Video recording
        self.is_recording = False
        self.video_writer = None
        self.motion_frames_buffer = deque(maxlen=30)
        
    def detect_motion(self, frame, background):
        """Phát hiện chuyển động trong khung hình so với nền"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_size, 0)

        # Tính hiệu giữa khung hình hiện tại và nền
        frame_delta = cv2.absdiff(background, gray)
        thresh = cv2.threshold(frame_delta, self.delta_threshold, 255, cv2.THRESH_BINARY)[1]

        # Mở rộng ngưỡng để lấp đầy các lỗ
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Tìm các đường viền
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_contours = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            motion_contours.append(contour)

        return motion_contours, thresh
    
    def update_background(self, frame, alpha=0.01):
        """Cập nhật nền động để thích nghi với môi trường"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_size, 0)
        
        if self.background is None:
            self.background = gray.copy().astype("float")
        else:
            cv2.accumulateWeighted(gray, self.background, alpha)
    
    def draw_info(self, frame, contours, motion_detected, fps=0):
        """Vẽ thông tin lên khung hình"""
        height, width = frame.shape[:2]
        
        # Vẽ các hộp giới hạn và thông tin cho mỗi đối tượng
        for i, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # Vẽ hộp
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Tính diện tích
            area = cv2.contourArea(contour)
            
            # Vẽ nhãn
            label = f"Obj{i+1}: {area:.0f}px"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Vẽ trọng tâm
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        
        # Tạo overlay cho thông tin
        overlay = frame.copy()
        
        # Vẽ thanh trạng thái phía trên
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Thông tin trạng thái
        status_text = "⚠ CHUYỂN ĐỘNG PHÁT HIỆN" if motion_detected else "✓ Không có chuyển động"
        status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
        
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Thời gian: {timestamp}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Thông tin chi tiết
        cv2.putText(frame, f"Đối tượng: {len(contours)}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Tổng số lần phát hiện
        cv2.putText(frame, f"Tổng phát hiện: {self.total_motion_detected}", (width - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Chế độ ghi
        if self.is_recording:
            cv2.circle(frame, (width - 30, 60), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (width - 70, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Hướng dẫn phím
        info_y = height - 80
        cv2.putText(frame, "Phím tắt: [Q]Thoát [R]Ghi [S]Chụp [B]Reset nền [+/-]Độ nhạy",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def start_recording(self, frame_shape, fps=20.0):
        """Bắt đầu ghi video"""
        if not self.is_recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.video_dir, f"motion_{timestamp}.avi")
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, fps,
                (frame_shape[1], frame_shape[0])
            )
            self.is_recording = True
            print(f"▶ Bắt đầu ghi: {filename}")
            return filename
        return None
    
    def stop_recording(self):
        """Dừng ghi video"""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print("⏹ Đã dừng ghi video")
    
    def save_snapshot(self, frame):
        """Lưu ảnh chụp màn hình"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.image_dir, f"snapshot_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"📸 Đã lưu ảnh: {filename}")
        return filename

def main():
    """Hàm chính"""
    print("=" * 60)
    print("HỆ THỐNG PHÁT HIỆN CHUYỂN ĐỘNG")
    print("=" * 60)
    print("Phím tắt:")
    print("  [Q] - Thoát chương trình")
    print("  [R] - Bật/Tắt ghi video")
    print("  [S] - Chụp ảnh màn hình")
    print("  [B] - Reset nền (cập nhật nền mới)")
    print("  [+] - Tăng độ nhạy (giảm ngưỡng)")
    print("  [-] - Giảm độ nhạy (tăng ngưỡng)")
    print("  [A] - Tăng diện tích tối thiểu")
    print("  [D] - Giảm diện tích tối thiểu")
    print("=" * 60)
    
    # Khởi tạo camera
    print("\n🎥 Đang khởi động camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Lỗi: Không thể mở camera!")
        return
    
    # Thiết lập camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera đã sẵn sàng!")
    time.sleep(2.0)
    
    # Khởi tạo detector
    detector = MotionDetector(min_area=500, delta_threshold=25)
    
    # Lấy khung hình nền ban đầu
    print("📷 Đang chụp ảnh nền...")
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc từ camera!")
        cap.release()
        return
    
    detector.background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector.background = cv2.GaussianBlur(detector.background, (21, 21), 0).astype("float")
    print("✓ Đã thiết lập nền!\n")
    
    # Biến đếm
    frame_count = 0
    start_time = time.time()
    auto_record = False
    motion_detected_last = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Không thể đọc frame!")
                break
            
            frame_count += 1
            
            # Phát hiện chuyển động
            motion_contours, thresh = detector.detect_motion(frame, detector.background.astype("uint8"))
            motion_detected = len(motion_contours) > 0
            
            # Đếm chuyển động
            if motion_detected and not motion_detected_last:
                detector.total_motion_detected += 1
            motion_detected_last = motion_detected
            
            # Cập nhật nền động
            detector.update_background(frame, alpha=0.01)
            
            # Tính FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Vẽ thông tin
            display_frame = detector.draw_info(frame.copy(), motion_contours, motion_detected, fps)
            
            # Auto recording
            if auto_record:
                if motion_detected and not detector.is_recording:
                    detector.start_recording(frame.shape)
                
                if detector.is_recording:
                    detector.video_writer.write(display_frame)
            
            # Hiển thị
            cv2.imshow("Motion Detection System - Press Q to quit", display_frame)
            
            # Hiển thị threshold (debug)
            if motion_detected:
                cv2.imshow("Threshold (Debug)", cv2.resize(thresh, (320, 240)))
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n👋 Đang thoát...")
                break
                
            elif key == ord('r') or key == ord('R'):
                auto_record = not auto_record
                if auto_record:
                    print("🔴 Chế độ tự động ghi: BẬT")
                else:
                    print("⚪ Chế độ tự động ghi: TẮT")
                    detector.stop_recording()
                    
            elif key == ord('s') or key == ord('S'):
                detector.save_snapshot(display_frame)
                
            elif key == ord('b') or key == ord('B'):
                detector.background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detector.background = cv2.GaussianBlur(detector.background, (21, 21), 0).astype("float")
                print("🔄 Đã reset nền!")
                
            elif key == ord('+') or key == ord('='):
                detector.delta_threshold = max(5, detector.delta_threshold - 5)
                print(f"⬆ Độ nhạy tăng (Ngưỡng: {detector.delta_threshold})")
                
            elif key == ord('-') or key == ord('_'):
                detector.delta_threshold = min(100, detector.delta_threshold + 5)
                print(f"⬇ Độ nhạy giảm (Ngưỡng: {detector.delta_threshold})")
                
            elif key == ord('a') or key == ord('A'):
                detector.min_area = min(5000, detector.min_area + 100)
                print(f"📏 Diện tích tối thiểu: {detector.min_area}px")
                
            elif key == ord('d') or key == ord('D'):
                detector.min_area = max(100, detector.min_area - 100)
                print(f"📏 Diện tích tối thiểu: {detector.min_area}px")
    
    except KeyboardInterrupt:
        print("\n⚠ Ngắt bởi người dùng...")
    
    finally:
        # Dọn dẹp
        print("\n🧹 Đang dọn dẹp...")
        detector.stop_recording()
        cap.release()
        cv2.destroyAllWindows()
        
        # Thống kê
        print("\n" + "=" * 60)
        print("THỐNG KÊ")
        print("=" * 60)
        print(f"Tổng số frame: {frame_count}")
        print(f"FPS trung bình: {fps:.1f}")
        print(f"Tổng lần phát hiện chuyển động: {detector.total_motion_detected}")
        print(f"Thời gian chạy: {elapsed:.1f} giây")
        print("=" * 60)
        print("✓ Chương trình đã kết thúc.")

if __name__ == "__main__":
    main()