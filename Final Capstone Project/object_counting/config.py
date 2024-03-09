import cv2


def get_first_frame(src):
    cap = cv2.VideoCapture(src)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return None
    
    # Capture the first frame
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if ret:
        return frame
    
    return None


class PointSelector:
    def __init__(self, src, resized_width, resized_height,
                 window_name,
                 entry_line = None, exit_line = None):
        self.points = []

        image = get_first_frame(src=src)
        self.image = cv2.resize(image, (resized_width, resized_height))
        if entry_line is not None:
            cv2.line(self.image, entry_line[0], entry_line[-1], (0, 255, 0), 3)
        if exit_line is not None:
            cv2.line(self.image, exit_line[0], exit_line[-1], (128, 0, 128), 3)

        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_click)

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(self.image, f'({x}, {y})', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.imshow(self.window_name, self.image)

    def show(self):
        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cv2.destroyAllWindows()



        
class LineDrager:
    def __init__(self, src, resized_width, resized_height, window_name,
                 line_color = None,
                 entry_line = None, exit_line = None) -> None:
        
        self.line_color = line_color

        image = get_first_frame(src=src)
        self.image = cv2.resize(image, (resized_width, resized_height))
        if entry_line is not None:
            cv2.line(self.image, entry_line[0], entry_line[-1], (0, 255, 0), 3)
        if exit_line is not None:
            cv2.line(self.image, exit_line[0], exit_line[-1], (128, 0, 128), 3)

        self.window_name = window_name
        self.point1 = (-1, -1)
        self.point2 = (-1, -1)
        self.drawing = False
        self.mouse_x = None
        self.mouse_y = None

    # Hàm callback cho sự kiện kéo thả chuột
    def drag_and_drop(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing is False:
                self.point1 = (x, y)
                self.drawing = True
            else:
                self.point2 = (x, y)
                self.drawing = False

        self.mouse_x, self.mouse_y = x, y

        if self.drawing is True:
            image_copy = self.image.copy()
            cv2.line(image_copy, self.point1, (self.mouse_x, self.mouse_y), self.line_color, 3)
            cv2.imshow(self.window_name, image_copy)


    def show(self):
        # Sao chép hình ảnh để vẽ đường thẳng mà không ảnh hưởng đến hình ảnh gốc
        image_copy = self.image.copy()

        # Đặt tên cho cửa sổ OpenCV và gắn hàm callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.drag_and_drop)

        while True:
            # Hiển thị hình ảnh
            cv2.imshow(self.window_name, image_copy)

            # Vẽ đường thẳng từ point1 đến vị trí hiện tại của chuột
            if self.point1 != (-1, -1) and self.drawing is True:
                image_copy = self.image.copy()
                cv2.line(image_copy, self.point1, (self.mouse_x, self.mouse_y), self.line_color, 3)
                cv2.circle(image_copy, self.point1, 8, (0, 0, 255), -1)
                cv2.putText(image_copy, f'{self.point1}', self.point1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # Vẽ đường thẳng từ point1 đến point2
            if self.point2 != (-1, -1):
                cv2.line(image_copy, self.point1, self.point2, self.line_color, 3)
                cv2.circle(image_copy, self.point2, 8, (0, 0, 255), -1)
                cv2.putText(image_copy, f'{self.point2}', self.point2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # Thoát khỏi vòng lặp nếu người dùng nhấn phím 'esc'
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Giải phóng bộ nhớ và đóng cửa sổ
        cv2.destroyAllWindows()




def open_config(src, resized_width, resized_height):
    entry_line_drager = LineDrager(src=src, resized_width=resized_width, resized_height=resized_height,
                            window_name='Drag the Entry line: (Press ESC when done!)',
                            line_color=(0, 255, 0))
    entry_line_drager.show()
    entry_line = [entry_line_drager.point1, entry_line_drager.point2]

    sample_inside_point_selector = PointSelector(src=src, resized_width=resized_width, resized_height=resized_height,
                                                 window_name='Choose inside zone: (Press ESC when done!)',
                                                 entry_line=entry_line)
    sample_inside_point_selector.show()
    sample_inside_point = sample_inside_point_selector.points[-1]

    exit_line_drager = LineDrager(src=src, resized_width=resized_width, resized_height=resized_height,
                           window_name='Drag the Exit line: (Press ESC when done!)',
                           line_color=(128, 0, 128),
                           entry_line=entry_line)
    exit_line_drager.show()
    exit_line = [exit_line_drager.point1, exit_line_drager.point2]

    sample_outside_point_selector = PointSelector(src=src, resized_width=resized_width, resized_height=resized_height,
                                                  window_name='Choose outside zone: (Press ESC when done!)',
                                                  entry_line=entry_line, exit_line=exit_line)
    sample_outside_point_selector.show()
    sample_outside_point = sample_outside_point_selector.points[-1]


    return entry_line, exit_line, sample_inside_point, sample_outside_point