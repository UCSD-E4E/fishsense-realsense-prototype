##this is for reading bags, changing the bag file from 1.bag to whatever u want

# python detect_bag.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4

import os
import pyrealsense2 as rs
import math

# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('bag', './data/video/1.bag', 'path to input bag or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.bag
    # get video name by using split method
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin bag capture
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, "1.bag")
        config.enable_all_streams()
        pipeline.start(config)
    finally:
        pass

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(640)
        height = int(480)
        fps = int(15)
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    align_to = rs.stream.color
    align = rs.align(align_to)
    for i in range(10):
        pipeline.wait_for_frames()
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        if frames is not None:
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            # depth_color_frame = rs.colorizer().colorize(depth_frame)

            frame = np.asanyarray(color_frame.get_data())
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to allow detections for only people)
        # allowed_classes = ['person']

        counted_things = (count_objects(pred_bbox, by_class=True, allowed_classes=allowed_classes))
        classesfound = []
        for i in range(len(class_names)):
            classesfound.append(class_names[i])
        fishes = []
        if ('fish' in counted_things):
            fishes_found = counted_things['fish']

            for i in range(fishes_found):
                fishes.append([bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]])

        # if crop flag is enabled, crop each detection and save it as new image
        if FLAGS.crop:
            crop_rate = 150  # capture images every so many frames (ex. crop photos every 150 frames)
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_path)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            if frame_num % crop_rate == 0:
                final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
                try:
                    os.mkdir(final_path)
                except FileExistsError:
                    pass
                crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
            else:
                pass

        if FLAGS.count:
            # count objects found
            counted_classes = count_objects(pred_bbox, by_class=False, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes,
                                    read_plate=FLAGS.plate)
        else:
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes,
                                    read_plate=FLAGS.plate)

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(fishes)

        delta = .1
        for i in range(len(fishes)):

            xmin = int(fishes[i][0])
            ymin = int(fishes[i][1])
            xmax = int(fishes[i][2])
            ymax = int(fishes[i][3])
            depth_list = []
            depth_dic = {}
            # fish = []
            boundary = []

            # Go over every single pixel in the bounding box.
            # Gather depth data from each pixel, and store it all in a list.
            # Create a dictionary to easily map from the median x,y coordinate later.
            for x in range(xmin, xmax):
                for y in range(ymin, ymax):
                    # This gets the depth value at a certain coordinate on the depth frame
                    depth_val = round(depth_frame.get_distance(int(x), int(y)), 4)

                    depth_list.append(depth_val)
                    depth_dic[depth_val] = (x, y, depth_val, depth_val)

            median = statistics.median_low(depth_list)
            median_pixel = depth_dic[median]

            # Do a BFS search to see which pixels are fish and which are not fish
            queue = [median_pixel]
            visited = set()

            # deltas = []

            look_for_delta = True

            while len(queue) > 0:

                curr_pixel = queue.pop()
                x, y, depth, parent_depth = curr_pixel
                visited.add((x, y))

                # if look_for_delta:
                #     look_for_delta = False
                #     deltas.append(abs(depth_frame.get_distance(int(x), int(y-1)) - depth))
                #     deltas.append(abs(depth_frame.get_distance(int(x), int(y + 1)) - depth))
                #     deltas.append(abs(depth_frame.get_distance(int(x-1), int(y)) - depth))
                #     deltas.append(abs(depth_frame.get_distance(int(x+1), int(y)) - depth))

                if abs(depth - parent_depth) < delta:
                    # fish.append((x, y))
                    if (y - 1 >= ymin) and (x, y - 1) not in visited:
                        queue.append((x, y - 1, depth_frame.get_distance(int(x), int(y - 1)), depth))

                    if (y + 1 <= ymax) and (x, y + 1) not in visited:
                        queue.append((x, y + 1, depth_frame.get_distance(int(x), int(y + 1)), depth))

                    if (x - 1 >= xmin) and (x - 1, y) not in visited:
                        queue.append((x - 1, y, depth_frame.get_distance(int(x - 1), int(y)), depth))

                    if (x + 1 <= xmax) and (x + 1, y) not in visited:
                        queue.append((x + 1, y, depth_frame.get_distance(int(x + 1), int(y)), depth))

                # Check up, down, left, and right pixels, if these pixels are within range of their parent.
                else:
                    boundary.append((x, y))
            for x, y in boundary:
                result[y, x] = (255, 0, 0)
                # max_delta = max(deltas)
                #
                # if max_delta > .00001:
                #     print(max(deltas))

            # Used to store the two different tuples that are the farthest apart, as well as the distance between them.
            distances = {}
            checked = set()

            for x1, y1 in boundary:

                for x2, y2 in boundary:

                    if (x1, y1) != (x2, y2) and ((x2, y2), (x1, y1)) not in checked:
                        checked.add(((x1, y1), (x2, y2)))
                        # find the depth at x1, y1, and cast it to a point in 3d space
                        origin_dist = depth_frame.get_distance(int(x1), int(y1))
                        point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [int(x1), int(y1)], origin_dist)

                        # find the depth at x2, y2, and cast it to a point in 3d space
                        new_dist = depth_frame.get_distance(int(x2), int(y2))
                        point2 = rs.rs2_deproject_pixel_to_point(color_intrin, [int(x2), int(y2)], new_dist)

                        # Calculate distance
                        dist = round(math.sqrt(
                            math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2) + math.pow(
                                point1[2] - point2[2], 2)), 5)

                        distances[dist] = [(x1, y1), (x2, y2)]

            max_distance = max(distances.keys())

            [max_point_1, max_point_2] = distances[max_distance]

            cv2.putText(result, "{} cm".format(max_distance * 100), (fishes[i][2], fishes[i][1]), color=(255, 0, 0),
                        thickness=2, fontFace=1, fontScale=2)
            cv2.line(result, max_point_1, max_point_2, (0, 255, 0), 4)

        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
