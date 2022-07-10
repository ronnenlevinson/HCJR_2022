import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from graphics_utility import convert_pixel_coordinates_to_normalized_coordinates, draw_polygon_on_image, \
    is_polygon_A_inside_polygon_B
from config import FILENAME_DATETIME_FORMAT_GLOBAL, EXTRA_SCENE_DATA_DIR_GLOBAL


def get_normalized_polygon_ROIs(
        roi_helper,
        image,
        n_polygons,
        max_tries=10
):
    """
    Let user input one or more region-of-interest (ROI) polygons by drawing connected lines over an image.
    """
    # The EasyROI package has (or had) a known problem with that can throw an exception during or
    # just after drawing a polygon. The error goes away when retried.
    tries = 0
    ROIs_drawn = False
    while not ROIs_drawn:
        try:
            tries += 1
            polygon_roi = roi_helper.draw_polygon(image, n_polygons)
            ROIs_drawn = True
        except Exception as err:
            print('Exception = ', err)
            if tries > max_tries:
                return None
    if len(polygon_roi['roi']) < n_polygons:
        return None

    pixel_polygons = {i:
                          np.array(polygon_roi['roi'][i]['vertices']) for i in range(n_polygons)
                      }
    # Convert the pixel polygons returned by EasyROI to normalized polygons.
    normalized_polygons = {
        k: convert_pixel_coordinates_to_normalized_coordinates(image=image, pixel_points=v) \
        for k, v in pixel_polygons.items()
    }
    return normalized_polygons


def draw_roi_polygons_on_image(image, polygons, thickness=1):
    """
    Draw ROI polygons on an image, colored coding them according to region type.
    """
    if polygons is None:
        return image
    for region, polygon in polygons.items():
        if region == 'inner_PTR':
            colorname = 'green'
        elif region == 'outer_PTR':
            colorname = 'orange'
        elif region == 'ATR':
            colorname = 'blue'
        draw_polygon_on_image(image=image, polygon=polygon, colorname=colorname, thickness=thickness)
    return image


def get_PTR_borders(roi_helper, image):
    """
    Let user input two polygons identifying the inner and outer borders of the PTR.
    If one polygon is inside the other, return them as the inner and outer borders of the PTR; otherwise, return None.
    """
    polygons = get_normalized_polygon_ROIs(roi_helper=roi_helper, image=image, n_polygons=2)
    if polygons is None:
        return None
    polygon0_in_polygon1 = is_polygon_A_inside_polygon_B(polygon_A=polygons[0], polygon_B=polygons[1], image=image)
    polygon1_in_polygon0 = is_polygon_A_inside_polygon_B(polygon_A=polygons[1], polygon_B=polygons[0], image=image)
    if polygon0_in_polygon1:
        inner_PTR, outer_PTR = polygons[0], polygons[1]
    elif polygon1_in_polygon0:
        inner_PTR, outer_PTR = polygons[1], polygons[0]
    else:
        return None
    PTR_borders = dict(inner_PTR=inner_PTR, outer_PTR=outer_PTR)
    return PTR_borders


def get_ATR_border(roi_helper, image):
    """
    Let user input one polygon identifying the border of the heated region of the ATR, then return it as the
    border of the ATR.
    """
    polygons = get_normalized_polygon_ROIs(roi_helper=roi_helper, image=image, n_polygons=1)
    if polygons is None:
        return None
    ATR_border = dict(ATR=polygons[0])
    return ATR_border


def get_PTR_and_ATR_borders(roi_helper, image):
    """
    Let user input three polygons identifying the inner and outer borders of the PTR and the border of the
    heated region of the ATR. If exactly one polygon is inside another polygon, return them as the inner
    and outer borders of the PTR, and return the third polygon as the ATM border. Otherwise, return None.
    """
    polygons = get_normalized_polygon_ROIs(roi_helper=roi_helper, image=image, n_polygons=3)
    if polygons is None:
        return None
    polygon0_in_polygon1 = is_polygon_A_inside_polygon_B(polygon_A=polygons[0], polygon_B=polygons[1], image=image)
    polygon0_in_polygon2 = is_polygon_A_inside_polygon_B(polygon_A=polygons[0], polygon_B=polygons[2], image=image)
    polygon1_in_polygon0 = is_polygon_A_inside_polygon_B(polygon_A=polygons[1], polygon_B=polygons[0], image=image)
    polygon1_in_polygon2 = is_polygon_A_inside_polygon_B(polygon_A=polygons[1], polygon_B=polygons[2], image=image)
    polygon2_in_polygon0 = is_polygon_A_inside_polygon_B(polygon_A=polygons[2], polygon_B=polygons[0], image=image)
    polygon2_in_polygon1 = is_polygon_A_inside_polygon_B(polygon_A=polygons[2], polygon_B=polygons[1], image=image)
    if polygon0_in_polygon1 and not polygon0_in_polygon2:
        inner_PTR, outer_PTR, ATR = polygons[0], polygons[1], polygons[2]
    elif polygon0_in_polygon2 and not polygon0_in_polygon1:
        inner_PTR, outer_PTR, ATR = polygons[0], polygons[2], polygons[1]
    elif polygon1_in_polygon0 and not polygon1_in_polygon2:
        inner_PTR, outer_PTR, ATR = polygons[1], polygons[0], polygons[2]
    elif polygon1_in_polygon2 and not polygon1_in_polygon0:
        inner_PTR, outer_PTR, ATR = polygons[1], polygons[2], polygons[0]
    elif polygon2_in_polygon0 and not polygon2_in_polygon1:
        inner_PTR, outer_PTR, ATR = polygons[2], polygons[0], polygons[1]
    elif polygon2_in_polygon1 and not polygon2_in_polygon0:
        inner_PTR, outer_PTR, ATR = polygons[2], polygons[1], polygons[0]
    else:
        return None
    PTR_and_ATR_borders = dict(inner_PTR=inner_PTR, outer_PTR=outer_PTR, ATR=ATR)
    return PTR_and_ATR_borders


def save_roi_dict(roi_dict):
    """
    Save the dictionary of region-of-interest (ROI) polygons to a pickle file.
    """
    roi_dict_sorted = sort_dictionary(roi_dict)
    datetime_string = datetime.now().strftime(FILENAME_DATETIME_FORMAT_GLOBAL)
    roi_dict_folder = EXTRA_SCENE_DATA_DIR_GLOBAL
    roi_dict_pathname = os.path.join(roi_dict_folder, 'roi_dict.pickle')
    if os.path.exists(roi_dict_pathname):
        roi_dict_backup_folder = os.path.join(roi_dict_folder, 'ROI Dictionary Archive')
        roi_dict_backup_pathname = os.path.join(roi_dict_backup_folder, f'roi_dict_backup_{datetime_string}.pickle')
        os.rename(roi_dict_pathname, roi_dict_backup_pathname)
        print(f'\nRenamed {roi_dict_pathname} to {roi_dict_backup_pathname}')
    with open(roi_dict_pathname, 'wb') as f:
        pickle.dump(roi_dict_sorted, f)
        print(f'\nWrote {roi_dict_pathname}')


def load_roi_dict(filename='roi_dict.pickle'):
    """
    Load the dictionary of region-of-interest (ROI) polygons from a pickle file.
    """
    roi_dict_pathname = os.path.join(EXTRA_SCENE_DATA_DIR_GLOBAL, filename)
    if not os.path.exists(roi_dict_pathname):
        return None
    with open(roi_dict_pathname, 'rb') as f:
        roi_dict = pickle.load(f)
    return roi_dict


def find_most_recent_roi_polygons(scene_datetime_object, roi_dict):
    """
    Find for a given scene datetime the most recent ROI polygons in the ROI polygon dictionary.
    """
    if scene_datetime_object is None:
        return None
    roi_datetime_objects = pd.Series(
        [datetime.strptime(datetime_string, FILENAME_DATETIME_FORMAT_GLOBAL) for datetime_string in roi_dict.keys()])
    datetime_objects_to_now = roi_datetime_objects[roi_datetime_objects <= scene_datetime_object]
    if len(datetime_objects_to_now) == 0:
        return None
    most_recent_datetime_object = max(datetime_objects_to_now)
    most_recent_datetime_string = most_recent_datetime_object.strftime(FILENAME_DATETIME_FORMAT_GLOBAL)
    polygons = roi_dict[most_recent_datetime_string]
    return polygons


def sort_dictionary(original_dict):
    """
    Sort dictionary to list its keys in ascending order.
    """
    sorted_keys = sorted(original_dict.keys())
    sorted_dict = {k: original_dict[k] for k in sorted_keys}
    return sorted_dict
