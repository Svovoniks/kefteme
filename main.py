import os
import cv2
import time
import torch
import argparse
import numpy as np
import datetime

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

from config_manager import Config
from notification_center import Notifier

from multiprocessing import Process, Pipe

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


def start_detection(cam_source, pipe, config):
    detection_input_size = 384
    pose_input_size = '224x160'
    pose_backbone = 'resnet50'
    show_skeleton = True
    device = config['GENERAL']['device']

    notifier = Notifier(config['SOURCES'][cam_source], config)
    # DETECTION MODEL.
    inp_dets = detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG(device=device)

    resize_fn = ResizePadding(inp_dets, inp_dets)

    def preproc(image):
        """preprocess function for CameraLoader.
        """
        image = resize_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    fps_time = 0
    f = 0
    try:
        while cam.grabbed():
            f += 1
            frame = cam.getitem()
            image = frame.copy()

            # Detect humans bbox in the frame with detector model.
            detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

            # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
            tracker.predict()
            # Merge two source of predicted bbox together.
            for track in tracker.tracks:
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det

            detections = []  # List of Detections object for tracking.
            if detected is not None:
                #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
                # Predict skeleton pose of each bboxs.
                poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

                # Create Detections object.
                detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                        np.concatenate((ps['keypoints'].numpy(),
                                                        ps['kp_score'].numpy()), axis=1),
                                        ps['kp_score'].mean().numpy()) for ps in poses]

            # Update tracks by matching each track information of current and previous frame or
            # create a new track if no matched.
            try:
                tracker.update(detections)
            except Exception as e:
                notifier.handle_action('Fall Down', None)
                continue


            # Predict Actions of each track.
            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                center = track.get_center().astype(int)

                action = 'pending..'
                clr = (0, 255, 0)
                # Use 30 frames time-steps to prediction.
                if len(track.keypoints_list) == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = action_model.predict(pts, frame.shape[:2])
                    action_name = action_model.class_names[out[0].argmax()]
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)

                    image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)

                    notifier.handle_action(action_name, image)

                    if action_name == 'Fall Down':
                        clr = (255, 0, 0)
                    elif action_name == 'Lying Down':
                        clr = (255, 200, 0)

                # VISUALIZE.
                if track.time_since_update == 0:
                    if show_skeleton:
                        frame = draw_single(frame, track.keypoints_list[-1])
                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                    frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                        0.4, (255, 0, 0), 2)
                    frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                        0.4, clr, 1)

            # Show Frame.
            frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
            # frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
            #                     (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frame = cv2.putText(frame, str(datetime.datetime.now()),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frame = frame[:, :, ::-1]
            fps_time = time.time()

            pipe.send(frame)
    except Exception as e:
        print(e)
    finally:
        cv2.destroyAllWindows()
        cam.stop()    

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def show(pipes, grid_width=4, item_height=300):
    while True:
        images = [image_resize(k.recv(), height=item_height) for k in pipes]
        blank_image = np.zeros((*images[0].shape[:2],3), np.uint8)
        images += [blank_image]*(grid_width-len(images)%grid_width)
        rows = []
        for i in range(len(images)//grid_width):
            rows.append(np.concatenate(images[i*grid_width : (i+1)*grid_width], axis=1))

        image = np.concatenate(rows, axis=1)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def start(config):
    procs = []
    pipes = []
    for source in config['SOURCES']:
        a, b = Pipe(duplex=True)
        p = Process(target=start_detection, args=(source, a, config))
        pipes.append(b)
        procs.append(p)
        p.start()

    show(pipes, int(config['GENERAL']['grid_width']), int(config['GENERAL']['video_height']),)

    for i in pipes:
        i.close()

    for i in procs:
        i.join()

if __name__ == '__main__':
    config = Config()
    config.validate()
    start(config.config)