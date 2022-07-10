import cv2
import numpy as np

from config import VIS_HEIGHT_GLOBAL


def reformat_points_2D_to_3D(pts):
    """
    Convert a 2D array of points [[x1,y1], ...] to a 3D array of points [[[x1,y1], ...]] for use in
    OpenCV transformation functions that expect the latter.
    """
    # Apply np.float32() to ensure that transformation routines receive a numpy array of float values rather than, say,
    # a list of integer values.
    return np.float32([pts])


def reformat_points_3D_to_2D(pts):
    """
    Convert a 3D array of points [[[x1,y1], ...]] to a 2D array of points [[x1,y1], ...] for use outside
    OpenCV transformation functions.
    """
    return np.float32(pts).reshape(-1, 2)


def get_homography(
        vis_keypoints,
        ir_keypoints,
        region='face',
        homography_method=0
):
    """
    Compute a homography (planar transformation) mapping N > 3 points in one image to N corresponding points
    in another image.
    """
    NullResult = dict(transformation_matrix=None, mask=None, condition_number=None)
    if region not in ['face', 'left_hand', 'right_hand', 'both_hands']:
        print(f'Unknown region: {region}')
        return NullResult
    landmarks_field = f'pose_{region}_landmark_points'
    vis_points = vis_keypoints[landmarks_field]
    ir_points = ir_keypoints[landmarks_field]
    if vis_points is None or ir_points is None:
        return NullResult
    source_pts = reformat_points_2D_to_3D(vis_points)
    dest_pts = reformat_points_2D_to_3D(ir_points)
    ransacReprojThreshold = 5 / VIS_HEIGHT_GLOBAL
    transformation_matrix, mask = \
        cv2.findHomography(
            srcPoints=source_pts,
            dstPoints=dest_pts,
            method=homography_method,
            ransacReprojThreshold=ransacReprojThreshold
        )
    condition_number = np.linalg.cond(transformation_matrix)
    result = dict(transformation_matrix=transformation_matrix, mask=mask, condition_number=condition_number)
    return result


def transform_polygon(polygon, transformation_matrix, label=None, verbose=False):
    """
    Transform a polygon by applying a transformation encoded in a matrix.
    """
    source_pts = reformat_points_2D_to_3D(polygon)
    if transformation_matrix is None:
        return None
    try:
        dest_pts = cv2.perspectiveTransform(source_pts, transformation_matrix)
        transformed_polygon = reformat_points_3D_to_2D(dest_pts)
    except:
        if verbose:
            print('cv2.perspectiveTransform failed')
            if label is not None:
                print(label)
            print('source_pts = ', source_pts)
            print('transformation_matrix = ', transformation_matrix)
        transformed_polygon = None

    return transformed_polygon


def transform_face_nose_eyeglasses_hand_palm_polygons(
        vis_keypoints,
        ir_keypoints,
        homography_method=0
):
    """
    Transform the face, nose, and eyeglasses polygons using the face homography and transform
    the left and right hands and palms using the both_hands homography.
    """
    regions = ['face', 'both_hands']
    homographies = {
        region: \
        get_homography(
            vis_keypoints=vis_keypoints,
            ir_keypoints=ir_keypoints,
            region=region,
            homography_method=homography_method
        ) for region in regions
    }

    ir_keypoints2 = ir_keypoints.copy()
    if homographies['face'] is None:
        ir_keypoints2['face_oval_polygon'] = None
        ir_keypoints2['nose_border_polygon'] = None
        ir_keypoints2['eyeglasses_border_polygon'] = None
    else:
        ir_keypoints2['face_oval_polygon'] = \
            transform_polygon(
                polygon=vis_keypoints['face_oval_polygon'],
                transformation_matrix=homographies['face']['transformation_matrix'],
                label='face_oval_polygon'
            )
        ir_keypoints2['nose_border_polygon'] = \
            transform_polygon(
                polygon=vis_keypoints['nose_border_polygon'],
                transformation_matrix=homographies['face']['transformation_matrix'],
                label='nose_border_polygon'
            )
        ir_keypoints2['eyeglasses_border_polygon'] = \
            transform_polygon(
                polygon=vis_keypoints['eyeglasses_border_polygon'],
                transformation_matrix=homographies['face']['transformation_matrix'],
                label='eyeglasses_border_polygon'
            )
    if homographies['both_hands'] is None:
        ir_keypoints2['left_hand_border_polygon'] = None
        ir_keypoints2['right_hand_border_polygon'] = None
        ir_keypoints2['left_palm_border_polygon'] = None
        ir_keypoints2['right_palm_border_polygon'] = None
    else:
        ir_keypoints2['left_hand_border_polygon'] = \
            transform_polygon(
                polygon=vis_keypoints['left_hand_border_polygon'],
                transformation_matrix=homographies['both_hands']['transformation_matrix'],
                label='left_hand_border_polygon'
            )
        ir_keypoints2['right_hand_border_polygon'] = \
            transform_polygon(
                polygon=vis_keypoints['right_hand_border_polygon'],
                transformation_matrix=homographies['both_hands']['transformation_matrix'],
                label='right_hand_border_polygon'
            )
        ir_keypoints2['left_palm_border_polygon'] = \
            transform_polygon(
                polygon=vis_keypoints['left_palm_border_polygon'],
                transformation_matrix=homographies['both_hands']['transformation_matrix'],
                label='left_palm_border_polygon'
            )
        ir_keypoints2['right_palm_border_polygon'] = \
            transform_polygon(
                polygon=vis_keypoints['right_palm_border_polygon'],
                transformation_matrix=homographies['both_hands']['transformation_matrix'],
                label='right_palm_border_polygon'
            )
    return ir_keypoints2, homographies


def cull_ir_hand_and_palm_polygons(
        vis_keypoints,
        ir_keypoints
):
    """
    Remove the transformed left/right hand/palm polygon from the IR image.
    Also remove pose-model left/right hand polygon from IR image
    if the hand-model left/right hand was not detected in the VIS image.
    """
    ir_keypoints_culled = ir_keypoints.copy()
    regions = [
        'left_hand_border_polygon',
        'right_hand_border_polygon',
        'left_palm_border_polygon',
        'right_palm_border_polygon'
    ]
    for region in regions:
        ir_keypoints_culled[region] = None
    if vis_keypoints['left_hand_border_polygon'] is None:
        ir_keypoints_culled['pose_left_hand_polygon'] = None
    if vis_keypoints['right_hand_border_polygon'] is None:
        ir_keypoints_culled['pose_right_hand_polygon'] = None
    return ir_keypoints_culled


def describe_condition_number(homography):
    """
    Return string describing condition number in one homography
    """
    if homography is None:
        return None
    k = homography['condition_number']
    if k is None:
        return None
    result = f'{k:.1f}'
    return result


def summarize_homographies(homographies):
    """
    Return string describing condition number in one or more homographies
    """
    strings = [f'{k}={describe_condition_number(v)}' for k, v in homographies.items()]
    result = f"transformation condition number: {', '.join(strings)}"
    return result


def flatten_transformation_matrix_to_semicolon_separated_string(matrix, digits=4):
    """
    Flatten 3 by 3 2D transformation matrix to a length-9 1D array, then return a string
    of the 1D array values in scientific notation, separated by semicolons.
    """
    if matrix is None:
        return None
    flattened_array = matrix.reshape(9)
    # To convert the following string back to the original 3 by 3 matrix, use
    # np.array(s.split(';')).reshape(3,3).astype(np.float32)
    s = ';'.join(f'{x:.{digits}e}' for x in flattened_array)
    return s
