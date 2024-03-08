import cv2

# Hàm callback cho sự kiện kéo thả chuột
def drag_and_drop(event, x, y, flags, param):
    global point1, point2, drawing, mouse_x, mouse_y

    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing is False:
            point1 = (x, y)
            drawing = True
        else:
            point2 = (x, y)
            drawing = False

    mouse_x, mouse_y = x, y

    if drawing is True:
        image_copy = image.copy()
        cv2.line(image_copy, point1, (mouse_x, mouse_y), (0, 255, 0), 3)
        cv2.imshow('Select Points', image_copy)

# Khởi tạo các biến
point1 = (-1, -1)
point2 = (-1, -1)
drawing = False

# Đọc hình ảnh
image = cv2.imread('example.jpg')
image = cv2.resize(image, (1280, 720))

# Sao chép hình ảnh để vẽ đường thẳng mà không ảnh hưởng đến hình ảnh gốc
image_copy = image.copy()

# Đặt tên cho cửa sổ OpenCV và gắn hàm callback
cv2.namedWindow('Select Points')
cv2.setMouseCallback('Select Points', drag_and_drop)

while True:
    # Hiển thị hình ảnh
    cv2.imshow('Select Points', image_copy)

    # Vẽ đường thẳng từ point1 đến vị trí hiện tại của chuột
    if point1 != (-1, -1) and drawing is True:
        image_copy = image.copy()
        cv2.line(image_copy, point1, (mouse_x, mouse_y), (0, 255, 0), 3)
        cv2.circle(image_copy, point1, 8, (0, 0, 255), -1)
        cv2.putText(image_copy, f'{point1}', point1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Vẽ đường thẳng từ point1 đến point2
    if point2 != (-1, -1):
        cv2.line(image_copy, point1, point2, (0, 255, 0), 3)
        cv2.circle(image_copy, point2, 8, (0, 0, 255), -1)
        cv2.putText(image_copy, f'{point2}', point2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Thoát khỏi vòng lặp nếu người dùng nhấn phím 'esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Giải phóng bộ nhớ và đóng cửa sổ
cv2.destroyAllWindows()
