import cv2
from ultralytics import YOLO
from loitering_detection_helper import Detector
from tqdm import tqdm


# ----------------- START OF Configs ---------------------
src = '../Demo/4088949254922987959.mp4'
write_output = True
show_results = True
yolo_model_path = 'yolov8s.pt'
max_time = 3
min_movement=200
yolo_threshold = 0.5

output_video_path = f'loitering_detection_' + str(src.replace('/', '_')) + '.mp4'
fps_tracking = 5
frs_skip = 5
# ----------------- END OF Configs -----------------


# INITIAL PARAMETERS AND VARIABLES

detector = Detector(yolo_model_path=yolo_model_path,
                    max_time=max_time,
                    min_movement=min_movement,
                    fps_tracking=fps_tracking,
                    yolo_threshold=yolo_threshold)

cap = cv2.VideoCapture(src)

fr_count = -1
prev_results = None


src_type = type(src)  # if int: live camera else: video

if src_type is str:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width, height = (1280, 720)

elif src_type is int:
    width, height = (1280, 720)


if write_output:
    # Define video output parameters
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))


if src_type is str:
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        fr_count += 1

        skip_fr = False

        if fr_count % frs_skip != 0:
            skip_fr = True

        # frame = cv2.resize(frame, (1280, 720))

        loiterings, current_people = detector.run(frame=frame, skip_fr=skip_fr, prev_results=prev_results)

        if show_results:
            cv2.imshow(f'Loitering Detection: {src}', frame)
            
            key = cv2.waitKey(10)
            if key == 27:
                break

        if write_output:
            out.write(frame)

        prev_results = {
            'loiterings': loiterings,
            'current_people': current_people
        }


elif src_type is int:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        fr_count += 1

        skip_fr = False

        if fr_count % frs_skip != 0:
            skip_fr = True

        frame = cv2.resize(frame, (width, height))

        loiterings, current_people = detector.run(frame=frame, skip_fr=skip_fr, prev_results=prev_results)

        if show_results:
            cv2.imshow(f'Loitering Detection: {src}', frame)
            
            key = cv2.waitKey(10)
            if key == 27:
                break

            elif key == 99:  # c pressed --> clear loiterings
                detector.clear(loiterings=loiterings)
                

        if write_output:
            out.write(frame)

        prev_results = {
            'loiterings': loiterings,
            'current_people': current_people
        }


        
cv2.destroyAllWindows()
out.release()
cap.release()