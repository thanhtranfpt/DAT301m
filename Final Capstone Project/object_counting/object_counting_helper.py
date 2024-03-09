from ultralytics import YOLO
import cv2
from collections import OrderedDict


class LimitedDict(OrderedDict):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        if len(self) >= self.max_size:
            # If the maximum number of pairs is reached, remove the oldest added pair
            self.popitem(last=False)
        super().__setitem__(key, value)



class Tracker:
    def __init__(self, yolo_model_path, threshold = 0.25, max_object_tracking = 1000,
                 max_movement_history = 120) -> None:
        self.yolo_model = YOLO(yolo_model_path)
        self.threshold = threshold
        self.movement_history = LimitedDict(max_size=max_object_tracking)
        self.max_movement_history = max_movement_history

    def get_current_objects(self, yolo_results, object_class = 0):
        current_objects = {}  #----- current_objects = {} ==> current_objects[f"{obj_id}"] = {"bbox": xywh, "conf": conf}
        negative_id = -1
        for box in yolo_results[0].boxes:
            cls = int(box.cls[0])
            if cls != object_class:
                continue
            conf = float(box.conf[0])
            if conf < self.threshold:
                continue
            if box.id is not None:
                obj_id = int(box.id[0])
            else:
                obj_id = negative_id
                negative_id -= 1
            xywh = [int(coor) for coor in box.xywh[0] ]
            current_objects[f"{obj_id}"] = {"bbox": xywh, "conf": conf}
            if obj_id < 0:
                continue

            if f"{obj_id}" in self.movement_history:
                self.movement_history[f"{obj_id}"].append(xywh)
            else:
                self.movement_history[f"{obj_id}"] = [xywh, xywh]
            if len(self.movement_history[f"{obj_id}"]) > self.max_movement_history:
                self.movement_history[f"{obj_id}"] = self.movement_history[f"{obj_id}"][1:]
            
            
        return current_objects
    
    def track(self, frame):
        yolo_results = self.yolo_model.track(frame, persist=True, verbose = False)
        current_people = self.get_current_objects(yolo_results=yolo_results, object_class=0)

        return current_people
    


class Counter:
    def __init__(self, yolo_model_path, yolo_threshold = 0.5, max_object_tracking = 1000, max_movement_history = 5,
                 entry_line = [(337, 586), (734, 498)],
                 exit_line = [(295, 655), (332, 717)],
                 sample_inside_point = (100, 200),
                 sample_outside_point = (500, 600)) -> None:
        
        self.entry_line = entry_line
        self.exit_line = exit_line
        self.sample_inside_point = sample_inside_point
        self.sample_outside_point = sample_outside_point

        self.list_went_in = set()
        self.list_went_out = set()

        self.tracker = Tracker(yolo_model_path=yolo_model_path, threshold=yolo_threshold, max_object_tracking=max_object_tracking, max_movement_history=max_movement_history)


    def run(self, frame, plot = True, skip_fr = False, prev_results = None):
        if skip_fr:
            current_people, list_go_in, list_go_out = prev_results['current_people'], prev_results['list_go_in'], prev_results['list_go_out']
            if plot:
                self.plot_results(list_go_in=list_go_in, list_go_out=list_go_out, frame=frame, current_people=current_people, is_prev_results=True)

            return prev_results
        

        current_people = self.tracker.track(frame=frame)
        list_go_in, list_go_out = self.update(current_people=current_people)
        self.plot_results(list_go_in=list_go_in, list_go_out=list_go_out, frame=frame, current_people=current_people)

        current_results = {
            'current_people': current_people,
            'list_go_in': list_go_in,
            'list_go_out': list_go_out
        }

        return current_results
    
    
    def update(self, current_people):
        movement_history = self.tracker.movement_history
        list_go_in = []
        list_go_out = []

        for person_id in current_people:
            if person_id not in movement_history:
                continue

            go_in = self.check_go_in(person_id=person_id)
            go_out = self.check_go_out(person_id=person_id)

            if go_in:
                self.list_went_in.add(person_id)
                list_go_in.append(person_id)
            if go_out:
                self.list_went_out.add(person_id)
                list_go_out.append(person_id)


        return list_go_in, list_go_out
    

    def get_bottom_midpoint(self, bbox):
        x_center, y_center, box_width, box_height = bbox
        # Calculate the midpoint of the bottom edge
        bottom_midpoint_x = x_center
        bottom_midpoint_y = y_center + box_height // 2
        return (bottom_midpoint_x, bottom_midpoint_y)
    

    def on_same_side(self, point_1, point_2, line_start = (200, 300), line_end = (500, 600)):
        x1, y1 = point_1
        x2, y2 = point_2
        x3, y3 = line_start
        x4, y4 = line_end

        # Calculate cross products
        cross_product_1 = (x4 - x3) * (y1 - y3) - (x1 - x3) * (y4 - y3)
        cross_product_2 = (x4 - x3) * (y2 - y3) - (x2 - x3) * (y4 - y3)

        # Check if cross products have the same sign
        return cross_product_1 * cross_product_2 > 0
    

    def check_go_in(self, person_id):
        sample_inside_point = self.sample_inside_point
        movement_history = self.tracker.movement_history
        if person_id not in movement_history:
            return False
        
        current_bbox = movement_history[person_id][-1]
        prev_bbox = movement_history[person_id][-2]
        current_position = self.get_bottom_midpoint(current_bbox)
        prev_position = self.get_bottom_midpoint(prev_bbox)
        if not self.on_same_side(current_position, prev_position, line_start=self.entry_line[0], line_end=self.entry_line[-1]):
            if self.on_same_side(current_position, sample_inside_point, line_start=self.entry_line[0], line_end=self.entry_line[-1]):
                return True
            
        return False
    
    def check_go_out(self, person_id):
        exit_line = self.exit_line
        sample_outside_point = self.sample_outside_point
        movement_history = self.tracker.movement_history
        if person_id not in movement_history:
            return False
        
        current_bbox = movement_history[person_id][-1]
        prev_bbox = movement_history[person_id][-2]
        current_position = self.get_bottom_midpoint(current_bbox)
        prev_position = self.get_bottom_midpoint(prev_bbox)

        if not self.on_same_side(current_position, prev_position, line_start=exit_line[0], line_end=exit_line[-1]):
            if self.on_same_side(current_position, sample_outside_point, line_start=exit_line[0], line_end=exit_line[-1]):
                return True
            
        return False
    

    def plot_results(self, list_go_in, list_go_out, frame, current_people,
                     is_prev_results = False,
                     text_background_color = (0, 0, 255),
                     text_size = 0.5, text_thickness = 1):
        '''
        Normal: yellow = (0, 255, 255)
        Go In: red = (0, 0, 255)
        Went In: green = (0, 255, 0)
        Go Out: blue = (255, 0, 0)
        Went Out: purple = (128, 0, 128)
        '''

        # Draw lines:
        cv2.line(img=frame, pt1=self.entry_line[0], pt2=self.entry_line[-1], color=(0, 255, 0), thickness=3)
        cv2.line(img=frame, pt1=self.exit_line[0], pt2=self.exit_line[-1], color=(128, 0, 128), thickness=3)

        # Plot bounding box
        
        for person_id, person_info in current_people.items():
            bbox, conf = person_info['bbox'], person_info['conf']

            x_center, y_center, box_width, box_height = bbox
            x_min = int(x_center - box_width / 2)
            y_min = int(y_center - box_height / 2)
            x_max = int(x_center + box_width / 2)
            y_max = int(y_center + box_height / 2)

            # Draw rectangle on the image
            color = (0, 255, 255)
            text_label = f'ID: {person_id} - conf: {conf:.2f}'
            if person_id in self.list_went_in:
                color = (0, 255, 0)
            if person_id in self.list_went_out:
                color = (128, 0, 128)
            if person_id in list_go_in:
                color = (0, 0, 255)
            if person_id in list_go_out:
                color = (255, 0, 0)

            if not is_prev_results:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

            # Add text with confidence score on a rectangle background
            (text_width, text_height), _ = cv2.getTextSize(text_label,
                                                           cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
            cv2.rectangle(frame, (x_min, y_min - text_height - 5), (x_min + text_width, y_min - 2),
                          text_background_color, cv2.FILLED)
            
            cv2.putText(frame, text_label, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), text_thickness, cv2.LINE_AA)
            

        # Plot number of went in, went out:
        NUM_IN = len(self.list_went_in)
        NUM_OUT = len(self.list_went_out)
        text_in = f'IN: {NUM_IN}'
        text_out = f'OUT: {NUM_OUT}'
        cv2.putText(frame, text_in, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, text_out, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)


        