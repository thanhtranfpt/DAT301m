import cv2
from object_counting_helper import Counter
from tqdm import tqdm
from config import open_config


# ----------------- START OF Configs ---------------------
src = '../Demo/TestVideo.avi'
write_output = True
show_results = True
yolo_model_path = '../models/yolov8s.pt'
yolo_threshold = 0.5

output_video_path = f'object_counting_' + str(src.replace('/', '_')) + '.mp4'
frs_skip = 1

# entry_line, exit_line, sample_inside_point, sample_outside_point = GET_BELOW
# ----------------- END OF Configs -----------------


# INITIAL PARAMETERS AND VARIABLES

cap = cv2.VideoCapture(src)

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



entry_line, exit_line, sample_inside_point, sample_outside_point = open_config(src=src, resized_width=width, resized_height=height)



counter = Counter(yolo_model_path=yolo_model_path, yolo_threshold=yolo_threshold,
                  entry_line=entry_line, exit_line=exit_line,
                  sample_inside_point=sample_inside_point, sample_outside_point=sample_outside_point)


fr_count = -1
prev_results = None

if src_type is str:
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        fr_count += 1

        skip_fr = False
        if fr_count % frs_skip != 0:
            skip_fr = True


        frame = cv2.resize(frame, (width, height))


        current_results = counter.run(frame=frame, skip_fr=skip_fr, prev_results=prev_results)

        if show_results:
            cv2.imshow(f'People Counting: {src}', frame)

            key = cv2.waitKey(10)
            if key == 27:
                break

        if write_output:
            out.write(frame)


        prev_results = current_results



elif src_type is int:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fr_count += 1

        skip_fr = False
        if fr_count % frs_skip != 0:
            skip_fr = True


        frame = cv2.resize(frame, (width, height))


        current_results = counter.run(frame=frame, skip_fr=skip_fr, prev_results=prev_results)

        if show_results:
            cv2.imshow(f'People Counting: {src}', frame)

            key = cv2.waitKey(10)
            if key == 27:
                break

        if write_output:
            out.write(frame)


        prev_results = current_results



cv2.destroyAllWindows()
out.release()
cap.release()