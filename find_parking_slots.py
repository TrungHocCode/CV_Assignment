import cv2
import json
import os
import numpy as np

# --- CẤU HÌNH ĐƯỜNG DẪN ---
IMAGE_PATH = 'first_frame.jpg'
JSON_PATH = 'parking_slots.json'

current_polygon = []
parking_slots = []
slot_id = 1
clone = None
img = None

def interpolate_points(p1, p2, n_slots):
    """Tính toán các điểm nội suy chia đều đoạn thẳng P1-P2 thành n_slots phần"""
    return [
        [int(p1[0] + (p2[0] - p1[0]) * i / n_slots), 
         int(p1[1] + (p2[1] - p1[1]) * i / n_slots)]
        for i in range(n_slots + 1)
    ]

def draw_polygon(event, x, y, flags, param):
    global current_polygon, parking_slots, clone, slot_id
    
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append([x, y])
        cv2.circle(clone, (x, y), 4, (0, 0, 255), -1)
        
        if len(current_polygon) > 1:
            cv2.line(clone, tuple(current_polygon[-2]), tuple(current_polygon[-1]), (0, 255, 0), 2)
            
        cv2.imshow("Dinh nghia Khong gian Bai do xe (Ban Tu Dong)", clone)
            
        # Khi click đủ 4 điểm của 1 HÀNG
        if len(current_polygon) == 4:
            cv2.line(clone, tuple(current_polygon[-1]), tuple(current_polygon[0]), (0, 255, 0), 2)
            cv2.imshow("Dinh nghia Khong gian Bai do xe (Ban Tu Dong)", clone)
            
            # --- YÊU CẦU NHẬP SỐ LƯỢNG Ô TỪ TERMINAL ---
            print("\n" + "="*40)
            print("ĐÃ VẼ XONG 1 HÀNG.")
            try:
                # Chuyển sự chú ý của người dùng về màn hình Terminal
                n_slots = int(input(">> Nhập số lượng ô đỗ có trong hàng này (ví dụ: 10): "))
                if n_slots <= 0:
                    raise ValueError
            except ValueError:
                print("[LỖI] Số lượng không hợp lệ. Đã hủy hàng vừa vẽ.")
                current_polygon = []
                # Khôi phục lại ảnh cũ chưa vẽ hàng lỗi
                refresh_screen()
                return

            # Nội suy chia lưới
            # Giả định thứ tự click: 1(Trái-trên) -> 2(Phải-trên) -> 3(Phải-dưới) -> 4(Trái-dưới)
            p1, p2, p3, p4 = current_polygon
            
            top_points = interpolate_points(p1, p2, n_slots)
            bottom_points = interpolate_points(p4, p3, n_slots) # Chú ý hướng từ p4 đến p3
            
            # Tạo các ô đỗ xe con
            for i in range(n_slots):
                # Lấy 4 góc của ô nhỏ hiện tại
                tl = top_points[i]
                tr = top_points[i+1]
                br = bottom_points[i+1]
                bl = bottom_points[i]
                
                sub_polygon = [tl, tr, br, bl]
                
                # Lưu vào danh sách tổng
                parking_slots.append({
                    "id": f"slot_{slot_id}",
                    "points": sub_polygon
                })
                
                slot_id += 1
            
            print(f"[THÀNH CÔNG] Đã chia thành {n_slots} ô nhỏ.")
            current_polygon = []
            refresh_screen() # Vẽ lại toàn bộ các ô đã có

def refresh_screen():
    """Hàm vẽ lại toàn bộ các ô đỗ xe đã lưu lên màn hình"""
    global clone
    clone = img.copy()
    
    for slot in parking_slots:
        pts = np.array(slot["points"], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(clone, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Tính tâm để ghi ID
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(clone, slot["id"].split('_')[1], (cX - 10, cY + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
    cv2.imshow("Dinh nghia Khong gian Bai do xe (Ban Tu Dong)", clone)

def main():
    global img, clone, parking_slots, current_polygon, slot_id
    
    if not os.path.exists(IMAGE_PATH):
        print(f"Lỗi: Không tìm thấy {IMAGE_PATH}.")
        return

    img = cv2.imread(IMAGE_PATH)
    clone = img.copy()

    cv2.namedWindow("Dinh nghia Khong gian Bai do xe (Ban Tu Dong)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dinh nghia Khong gian Bai do xe (Ban Tu Dong)", 1280, 720)
    cv2.setMouseCallback("Dinh nghia Khong gian Bai do xe (Ban Tu Dong)", draw_polygon)

    print("--- HƯỚNG DẪN CÁCH VẼ ĐỂ CHIA LƯỚI CHUẨN ---")
    print("Hãy click 4 điểm bao quanh 1 hàng xe theo thứ tự VÒNG TRÒN:")
    print("  1. Góc Phải-Trên")
    print("  2. Góc Phải-Dưới")
    print("  3. Góc Trái-Dưới")
    print("  4. Góc Trái-Trên")
    print("-> Sau khi click điểm thứ 4, quay lại màn hình Terminal để nhập số lượng ô!")
    print("\nBấm 's' để LƯU JSON. Bấm 'c' để XÓA LÀM LẠI. Bấm 'q' để THOÁT.")

    refresh_screen()

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if len(parking_slots) > 0:
                with open(JSON_PATH, 'w') as f:
                    json.dump({"parking_slots": parking_slots}, f, indent=4)
                print(f"\n[THÀNH CÔNG] Đã lưu tổng cộng {len(parking_slots)} ô vào {JSON_PATH}!")
            break
        elif key == ord('c'):
            parking_slots.clear()
            current_polygon.clear()
            slot_id = 1
            refresh_screen()
            print("\nĐã xóa toàn bộ.")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
