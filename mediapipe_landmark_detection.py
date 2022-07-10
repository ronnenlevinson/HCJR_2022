import cv2
import mediapipe as mp
import numpy as np

# 2022-06-05 Ronnen:
# mediapipe requires protobuf. Install protobuf 3.20.01 before installing mediapipe otherwise
# mediapipe will install protobuf 4.20.1, which breaks mediapipe.
# https://github.com/protocolbuffers/protobuf/issues/10051

# Notes about coordinate systems:
#
# matplotlib, OpenCV, and mediapipe use image (Cartesian) coordinates (x,y), where x is horizontal and y is vertical.
# In matplotlib, the origin (x=0, y=0) is at lower left and y increases upward.
# In OpenCV and mediapipe, the origin (x=0, y=0) is at upper left and y increases downward.
# matplotlib and OpenCV use pixel coordinates for raster images, such that x ranges from 0 to (width-1)
# and y ranges from 0 to (height-1). mediapipe uses normalized coordinates for raster images such that
# x and y each range from 0 to 1.
#
# Function cv2.imread() returns a 2-D numpy array of pixels with row/column coordinates (r,c),
# with (r=0, c=0) at upper left in the array. Since row is the first coordinate and column is the
# second coordinate in a 2-D numpy array, array coordinate (r=y, c=x) corresponds to image coordinate (x,y).


# The mediapipe detectors return the normalized coordinates (x,y) of each numbered landmark in the face, pose, and hards.
# Face mesh landmark map: <https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png>
# Pose landmark map: <https://developers.google.com/ml-kit/images/vision/pose-detection/landmarks-fixed.png>
# Hand landmark map: <https://google.github.io/mediapipe/images/mobile/hand_landmarks.png>
#
# For more information:
# Landmarks in face mesh: <https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py>
# Landmarks in pose: <https://google.github.io/mediapipe/solutions/pose.html>
# Landmarks in hands: <https://google.github.io/mediapipe/solutions/hands.html>
#
# A "connection" is a pair of landmark numbers joined to form part of a path. The mediapipe source code provides connection lists
# for some features, such as an oval bounding the face. For our purposes it is easier to work with a simple
# list of landmark numbers whose coordinates will serve as the vertices of a polygon.

def connections_to_landmark_numbers(connections):
    """
                Convert list of MediaPipe connections (connection = pair of landmark numbers) to a list of landmark numbers.
                For example, [ [0,1], [1,2], [2,3] ] would become [0, 1, 2].
                """
    landmark_numbers = [landmark_pair[0] for landmark_pair in connections]
    return landmark_numbers

def landmark_numbers_to_connections(landmarks):
    """
            Convert list of MediaPipe landmark numbers to a list of connections (connection = pair of landmark numbers).
            For example, [0, 1, 2] would become [ [0,1], [1,2] ] .
            """
    n = len(landmarks)
    connections = [(landmarks[i], landmarks[i + 1]) for i in range(n - 1)]
    return connections

# Connection list from <https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py>

FACEMESH_FACE_OVAL_CONNECTIONS = [
    (10, 338), (338, 297), (297, 332), (332, 284),
    (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400),
    (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234),
    (234, 127), (127, 162), (162, 21), (21, 54),
    (54, 103), (103, 67), (67, 109), (109, 10)]
FACEMESH_FACE_OVAL_LANDMARK_NUMBERS = connections_to_landmark_numbers(connections=FACEMESH_FACE_OVAL_CONNECTIONS)

# Nose border is based on Ronnen's reading of the face mesh landmark map.

FACEMESH_NOSE_BORDER_LANDMARK_NUMBERS = [6, 351, 412, 343, 437, 420, 360, 344, 438, 309, 250, 462, 370, 94,
                                         141, 242, 20, 79, 218, 115, 131, 198, 217, 114, 188, 122, 6]

# Based on Ronnen's reading of the hand landmark map.
# 2022-02-26: Revised palm border landmarks to include point 2 (THUMB MCP) for consistency
# with remaining palm border landmarks (e.g., 5 = index finger MPC, 9 = middle finger MCP, and so on).

HAND_PALM_BORDER_LANDMARK_NUMBERS = [0, 1, 2, 5, 9, 13, 17, 0]
HAND_ALL_BORDER_LANDMARK_NUMBERS = [0, 1, 2, 3, 4, 8, 12, 16, 20, 19, 18, 17, 0]

# Pose eye locations may be less accurate than facemesh eye locations.

POSE_LEFT_EYE_LANDMARK_NUMBER = 2
POSE_RIGHT_EYE_LANDMARK_NUMBER = 5
POSE_EYE_LANDMARK_NUMBERS = [POSE_LEFT_EYE_LANDMARK_NUMBER, POSE_RIGHT_EYE_LANDMARK_NUMBER]

# Pose nose location may be less accurate than facemesh nose location.

POSE_NOSE_LANDMARK_NUMBER = 0
POSE_LANDMARK_NUMBERS = {
    0: 'nose',
    1: 'left_eye_inner',
    2: 'left_eye',
    3: 'left_eye_outer',
    4: 'right_eye_inner',
    5: 'right_eye',
    6: 'right_eye_outer',
    7: 'left_ear',
    8: 'right_ear',
    9: 'mouth_left',
    10: 'mouth_right',
    11: 'left_shoulder',
    12: 'right_shoulder',
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
    17: 'left_pinky',
    18: 'right_pinky',
    19: 'left_index',
    20: 'right_index',
    21: 'left_thumb',
    22: 'right_thumb',
    23: 'left_hip',
    24: 'right_hip',
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle',
    29: 'left_heel',
    30: 'right_heel',
    31: 'left_foot_index',
    32: 'right_foot_index'
}
POSE_FACE_LANDMARK_NUMBERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
POSE_LEFT_ELBOW_LANDMARK_NUMBER = [13]
POSE_LEFT_HAND_LANDMARK_NUMBERS = [15, 17, 19, 21]
POSE_RIGHT_ELBOW_LANDMARK_NUMBER = [14]
POSE_RIGHT_HAND_LANDMARK_NUMBERS = [16, 18, 20, 22]
POSE_BOTH_HANDS_LANDMARK_NUMBERS = POSE_LEFT_HAND_LANDMARK_NUMBERS + POSE_RIGHT_HAND_LANDMARK_NUMBERS
FACEMESH_LEFT_EYE_CONNECTIONS = [(263, 249), (249, 390), (390, 373), (373, 374),
                                 (374, 380), (380, 381), (381, 382), (382, 362),
                                 (263, 466), (466, 388), (388, 387), (387, 386),
                                 (386, 385), (385, 384), (384, 398), (398, 362), (362, 263)]
FACEMESH_RIGHT_EYE_CONNECTIONS = [(33, 7), (7, 163), (163, 144), (144, 145),
                                  (145, 153), (153, 154), (154, 155), (155, 133),
                                  (33, 246), (246, 161), (161, 160), (160, 159),
                                  (159, 158), (158, 157), (157, 173), (173, 133), (133, 33)]
FACEMESH_LEFT_EYE_LANDMARK_NUMBERS = connections_to_landmark_numbers(connections=FACEMESH_LEFT_EYE_CONNECTIONS)
FACEMESH_RIGHT_EYE_LANDMARK_NUMBERS = connections_to_landmark_numbers(connections=FACEMESH_RIGHT_EYE_CONNECTIONS)
FACEMESH_EYEGLASSES_LANDMARK_NUMBERS = [35, 124, 46, 53, 52, 65, 55, 8, 285, 295, 282, 283,
                                        276, 353, 265, 261, 448, 449, 450, 451, 452, 453,
                                        464, 465, 351, 6, 122, 245, 244, 233, 232, 231, 230,
                                        229, 228, 31, 35]
mp_drawing = mp.solutions.drawing_utils  # MediaPipe drawing utilities
mp_holistic = mp.solutions.holistic  # MediaPipe function for creating holistic model (not the model itself)

# Avoid creating many instances of the model because each model consumes memory. It's best to create one instance
# of the trial model to process a series of images, then delete the model afterward.

def create_holistic_model(
        model_complexity=2,  # Complexity 0, 1, 2 range from fastest/least accurate to slowest/most accurate
        static_image_mode=True,  # True=process each frame separately; False=relate current frame to previous frame
        min_detection_confidence=0.50  # Scale 0 - 1
):
    """
    Create MediaPipe holistic model. Avoid creating multiple instances of the model because each instance consumes memory.
    """
    model = mp_holistic.Holistic(
        model_complexity=model_complexity,
        static_image_mode=static_image_mode,
        min_detection_confidence=min_detection_confidence
    )
    return model


def mediapipe_detection(image, model):
    """
    Convert blue-green-red (BGR) image loaded with cv2.imread() to red-green-blue (RGB), apply a mediapipe detector
    model, then return the original image and the model's results.
    """
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    # start_time = time.time()
    results = model.process(imageRGB)  # Make prediction
    # report_elapsed_time(start_time=start_time, digits=3)
    return image, results


def close_landmark_number_list(landmark_numbers):
    """
    Close list of landmark numbers by appending the first number to the list
    if the last number is not equal to the first number.
    """
    if isinstance(landmark_numbers, list):
        landmark_numbers_list = landmark_numbers
    else:
        landmark_numbers_list = [landmark_numbers]
    if len(landmark_numbers_list) > 2:
        first_landmark_number = landmark_numbers_list[0]
        last_landmark_number = landmark_numbers_list[-1]
        if last_landmark_number == first_landmark_number:
            landmark_numbers_closed = landmark_numbers_list
        else:
            landmark_numbers_closed = landmark_numbers_list + [first_landmark_number]
    return landmark_numbers_closed


def landmark_numbers_to_points(landmarks_object, landmark_numbers):
    """
    Return numpy array of float normalized [0-1] (x,y) coordinates of the points corresponding to landmark numbers.
    """
    if isinstance(landmark_numbers, list):
        landmark_numbers_list = landmark_numbers
    else:
        landmark_numbers_list = [landmark_numbers]
    points = np.array([(landmarks_object[i].x, landmarks_object[i].y) for i in landmark_numbers_list])
    return points


def landmark_numbers_to_polygon(landmarks_object, landmark_numbers):
    """
    Return numpy array of normalized [0-1] (x,y) coordinates of the polygon vertices corresponding to landmark numbers,
    first closing the landmark number list if necessary. A landmark polygon is similar to a landmark array but is
    guaranteed to be closed.
    """
    landmark_numbers_closed = close_landmark_number_list(landmark_numbers)
    vertices = landmark_numbers_to_points(
        landmarks_object=landmarks_object,
        landmark_numbers=landmark_numbers_closed
    )
    return vertices


def pose_landmark_numbers_to_visibilities(landmarks_object, landmark_numbers):
    """
    Return numpy array of visibilities [0-1] of pose landmark numbers.
    From https://google.github.io/mediapipe/solutions/pose.html :
    A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.
    """
    if isinstance(landmark_numbers, list):
        landmark_numbers_list = landmark_numbers
    else:
        landmark_numbers_list = [landmark_numbers]
    visibilities = np.array([landmarks_object[i].visibility for i in landmark_numbers_list])
    return visibilities


def get_face_oval_polygon_from_face_model(results):
    """
    Return list of normalized [0-1] (x,y) coordinates of the vertices in the face oval.
    """
    if results.face_landmarks is None:
        return None
    vertices = landmark_numbers_to_polygon(
        landmarks_object=results.face_landmarks.landmark,
        landmark_numbers=FACEMESH_FACE_OVAL_LANDMARK_NUMBERS
    )
    return vertices


def get_nose_border_polygon_from_face_model(results):
    """
    Return list of normalized [0-1] (x,y) coordinates of the vertices in the nose border.
    """
    if results.face_landmarks is None:
        return None
    vertices = landmark_numbers_to_polygon(
        landmarks_object=results.face_landmarks.landmark,
        landmark_numbers=FACEMESH_NOSE_BORDER_LANDMARK_NUMBERS
    )
    return vertices


def get_eye_border_polygon_from_face_model(results, eye='left'):
    """
    Return list of normalized [0-1] (x,y) coordinates of the vertices in the eye border.
    """
    if results.face_landmarks is None:
        return None
    if eye == 'left':
        eye_landmark_numbers = FACEMESH_LEFT_EYE_LANDMARK_NUMBERS
    elif eye == 'right':
        eye_landmark_numbers = FACEMESH_RIGHT_EYE_LANDMARK_NUMBERS
    vertices = landmark_numbers_to_polygon(
        landmarks_object=results.face_landmarks.landmark,
        landmark_numbers=eye_landmark_numbers
    )
    return vertices


def get_eyeglasses_border_polygon_from_face_model(results):
    """
    Return list of normalized [0-1] (x,y) coordinates of the vertices in the eyeglasses border.
    Relevant only of the subject is wearing eyeglasses!
    """
    if results.face_landmarks is None:
        return None
    vertices = landmark_numbers_to_polygon(
        landmarks_object=results.face_landmarks.landmark,
        landmark_numbers=FACEMESH_EYEGLASSES_LANDMARK_NUMBERS
    )
    return vertices


def get_hand_or_palm_border_polygon_from_hand_model(
        results,
        extent='hand',
        hand='left'
):
    """
    Return list of normalized [0-1] (x,y) coordinates of the vertices in the hand border. Extent is 'hand'
    for the entire hand, including fingers and possibly spaces between fingers, or 'palm' for just the palm.
    Extent 'hand' may overcapture the hand, while extent 'palm' may undercapture the hand.
    Hand is either 'left' or 'right'.
    """
    if hand == 'left' and results.left_hand_landmarks is None:
        return None
    if hand == 'right' and results.right_hand_landmarks is None:
        return None
    if extent == 'hand':
        hand_landmark_numbers = HAND_ALL_BORDER_LANDMARK_NUMBERS
    elif extent == 'palm':
        hand_landmark_numbers = HAND_PALM_BORDER_LANDMARK_NUMBERS
    if hand == 'left':
        landmarks_object = results.left_hand_landmarks.landmark
    elif hand == 'right':
        landmarks_object = results.right_hand_landmarks.landmark
    vertices = landmark_numbers_to_polygon(
        landmarks_object=landmarks_object,
        landmark_numbers=hand_landmark_numbers
    )
    return vertices


def get_pose_landmark_points_from_pose_model(results, landmark_numbers):
    """
    Return list of float normalized [0-1] (x,y) coordinates locating the centers
    of a given subset of the pose landmarks.
    """
    if results.pose_landmarks is None:
        return None
    points = landmark_numbers_to_points(
        landmarks_object=results.pose_landmarks.landmark,
        landmark_numbers=landmark_numbers)
    return points


def get_pose_landmark_polygon_from_pose_model(results, landmark_numbers):
    """
    Return polygon of float normalized [0-1] (x,y) coordinates locating the centers
    of a given subset of the pose landmarks.
    """
    if results.pose_landmarks is None:
        return None
    vertices = landmark_numbers_to_polygon(
        landmarks_object=results.pose_landmarks.landmark,
        landmark_numbers=landmark_numbers)
    return vertices


def get_pose_landmark_visibilities_from_pose_model(results, landmark_numbers):
    """
    Return numpy array of visibilities of pose landmarks.
    """
    if results.pose_landmarks is None:
        return None
    visibilities = pose_landmark_numbers_to_visibilities(
        landmarks_object=results.pose_landmarks.landmark,
        landmark_numbers=landmark_numbers)
    return visibilities


def draw_styled_landmarks(image, results):
    """
    Draw all face, hand, and pose landmarks returned by MediaPipe Holistic.
    """
    # This following is borrowed from some online code, probably on the Google MediaPipe github repository.

    # Draw face tesselations
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )
    # Draw face contours
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def get_feature_landmarks_and_polygons_from_image(image, model):
    """
    Apply MediaPipe Holistic model to image, then return dictionary of feature landmark points and polygons.
    """
    global my_results_global
    image2, results = mediapipe_detection(image=image, model=model)
    my_results_global = results
    # print(my_results_global)
    face_oval_polygon = get_face_oval_polygon_from_face_model(results=results)
    nose_border_polygon = get_nose_border_polygon_from_face_model(results=results)
    eyeglasses_border_polygon = get_eyeglasses_border_polygon_from_face_model(results=results)
    left_eye_border_polygon = get_eye_border_polygon_from_face_model(results=results, eye='left')
    right_eye_border_polygon = get_eye_border_polygon_from_face_model(results=results, eye='right')
    left_hand_border_polygon = \
        get_hand_or_palm_border_polygon_from_hand_model(results=results, extent='hand', hand='left')
    right_hand_border_polygon = \
        get_hand_or_palm_border_polygon_from_hand_model(results=results, extent='hand', hand='right')
    left_palm_border_polygon = \
        get_hand_or_palm_border_polygon_from_hand_model(results=results, extent='palm', hand='left')
    right_palm_border_polygon = \
        get_hand_or_palm_border_polygon_from_hand_model(results=results, extent='palm', hand='right')
    pose_left_hand_polygon = \
        get_pose_landmark_polygon_from_pose_model(results=results, landmark_numbers=POSE_LEFT_HAND_LANDMARK_NUMBERS)
    pose_right_hand_polygon = \
        get_pose_landmark_polygon_from_pose_model(results=results, landmark_numbers=POSE_RIGHT_HAND_LANDMARK_NUMBERS)
    pose_face_visibilities = \
        get_pose_landmark_visibilities_from_pose_model(results=results, landmark_numbers=POSE_FACE_LANDMARK_NUMBERS)
    pose_left_hand_visibilities = \
        get_pose_landmark_visibilities_from_pose_model(results=results, landmark_numbers=POSE_LEFT_HAND_LANDMARK_NUMBERS)
    pose_right_hand_visibilities = \
        get_pose_landmark_visibilities_from_pose_model(results=results, landmark_numbers=POSE_RIGHT_HAND_LANDMARK_NUMBERS)
    pose_face_landmark_points = \
        get_pose_landmark_points_from_pose_model(results=results, landmark_numbers=POSE_FACE_LANDMARK_NUMBERS)
    pose_eye_landmark_points = \
        get_pose_landmark_points_from_pose_model(results=results, landmark_numbers=POSE_EYE_LANDMARK_NUMBERS)
    pose_nose_landmark_point = \
        get_pose_landmark_points_from_pose_model(results=results, landmark_numbers=POSE_NOSE_LANDMARK_NUMBER)
    pose_left_hand_landmark_points = \
        get_pose_landmark_points_from_pose_model(results=results, landmark_numbers=POSE_LEFT_HAND_LANDMARK_NUMBERS)
    pose_right_hand_landmark_points = \
        get_pose_landmark_points_from_pose_model(results=results, landmark_numbers=POSE_RIGHT_HAND_LANDMARK_NUMBERS)
    pose_both_hands_landmark_points = \
        get_pose_landmark_points_from_pose_model(results=results, landmark_numbers=POSE_BOTH_HANDS_LANDMARK_NUMBERS)
    keypoints_dict = dict(
        face_oval_polygon=face_oval_polygon,
        nose_border_polygon=nose_border_polygon,
        eyeglasses_border_polygon=eyeglasses_border_polygon,
        left_eye_border_polygon=left_eye_border_polygon,
        right_eye_border_polygon=right_eye_border_polygon,
        left_hand_border_polygon=left_hand_border_polygon,
        right_hand_border_polygon=right_hand_border_polygon,
        left_palm_border_polygon=left_palm_border_polygon,
        right_palm_border_polygon=right_palm_border_polygon,
        pose_left_hand_polygon=pose_left_hand_polygon,
        pose_right_hand_polygon=pose_right_hand_polygon,
        pose_face_landmark_points=pose_face_landmark_points,
        pose_eye_landmark_points=pose_eye_landmark_points,
        pose_nose_landmark_point=pose_nose_landmark_point,
        pose_left_hand_landmark_points=pose_left_hand_landmark_points,
        pose_right_hand_landmark_points=pose_right_hand_landmark_points,
        pose_both_hands_landmark_points=pose_both_hands_landmark_points,
        pose_face_visibilities=pose_face_visibilities,
        pose_left_hand_visibilities=pose_left_hand_visibilities,
        pose_right_hand_visibilities=pose_right_hand_visibilities,
        results=results
    )
    return keypoints_dict
