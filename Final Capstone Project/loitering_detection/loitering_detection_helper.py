from ultralytics import YOLO
import cv2
from collections import OrderedDict
import math
import time


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
        self.start_time = LimitedDict(max_size=max_object_tracking)

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

            if not f'{obj_id}' in self.start_time:
                self.start_time[f'{obj_id}'] = time.time()
            
            
        return current_objects
    
    def track(self, frame):
        yolo_results = self.yolo_model.track(frame, persist=True, verbose = False)
        current_people = self.get_current_objects(yolo_results=yolo_results, object_class=0)

        return current_people



class Detector:
    def __init__(self, yolo_model_path, max_time = 60, min_movement = 300,
                 fps_tracking = 2,
                 yolo_threshold = 0.25, max_object_tracking = 1000) -> None:
        self.tracker = Tracker(yolo_model_path=yolo_model_path, threshold=yolo_threshold, max_object_tracking=max_object_tracking, max_movement_history=fps_tracking * max_time)
        self.max_time = max_time
        self.min_movement = min_movement


    def check_moving(self, person_id):
        movement_history = self.tracker.movement_history
        if person_id not in movement_history:
            return False
        
        positions = movement_history[person_id]
        distances = [self.distance(positions[i][:2], positions[i+1][:2]) for i in range(len(positions) - 1)]
        total_distance = sum(distances)

        if total_distance > self.min_movement:
            return True
        

    def check_too_long(self, person_id):
        start_time = self.tracker.start_time
        if person_id not in start_time:
            return False
        
        first = start_time[person_id]
        now = time.time()

        if (now - first) > self.max_time:
            return True
        
        return False


    def run(self, frame, plot = True, skip_fr = False, prev_results = None):
        if skip_fr:
            loiterings, current_people = prev_results['loiterings'], prev_results['current_people']
            if plot:
                self.plot_results(loiterings=loiterings, frame=frame, current_people=current_people, is_prev_results=True)
            return loiterings, current_people
        

        loiterings = []
        current_people = self.tracker.track(frame=frame)
        for person_id in current_people:
            for_too_long = self.check_too_long(person_id=person_id)
            if for_too_long:
                is_moving = self.check_moving(person_id=person_id)
                if is_moving:
                    loiterings.append(person_id)

        if plot:
            self.plot_results(loiterings=loiterings, frame=frame, current_people=current_people)

        return loiterings, current_people
    

    def clear(self, loiterings):
        for person_id in loiterings:
            self.tracker.movement_history.pop(key=person_id)
            self.tracker.start_time.pop(key=person_id)

            
    
    def plot_results(self, loiterings, frame, current_people, is_prev_results = False):
        for person_id, person_info in current_people.items():
            bbox = person_info['bbox']
            conf = person_info['conf']

            x_center, y_center, box_width, box_height = bbox
            x_min = int(x_center - box_width / 2)
            y_min = int(y_center - box_height / 2)
            x_max = int(x_center + box_width / 2)
            y_max = int(y_center + box_height / 2)

            if person_id in loiterings:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)


            if not is_prev_results:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

            # Add text with confidence score on a rectangle background
            text_background_color = color
            text_size, text_thickness = 0.5, 1
            (text_width, text_height), _ = cv2.getTextSize(f'ID: {person_id} Conf: {conf:.2f}',
                                                           cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
            cv2.rectangle(frame, (x_min, y_min - text_height - 5), (x_min + text_width, y_min - 2),
                          text_background_color, cv2.FILLED)
            
            cv2.putText(frame, f'ID: {person_id} Conf: {conf:.2f}', (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), text_thickness, cv2.LINE_AA)
            
        return frame
    

    def distance(self, point_1, point_2):
        x1, y1 = point_1
        x2, y2 = point_2
        
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    