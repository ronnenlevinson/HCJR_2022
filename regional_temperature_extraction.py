import numpy as np

from graphics_utility import extract_raster_array_values_inside_multiple_polygons


def prepare_regions_dictionary(keypoints_dict):
    """
    Create dictionary of polygons for the body regions of interest: face oval and nose border from face mesh model,
    left and right hands and palms from hands model, and left and right hands from pose model.
    """
    # Abbreviations for various Google MediaPipe models
    # F = Face Mesh
    # H = Hands
    # P = Pose
    mapping = dict(
        face_F='face_oval_polygon',
        nose_F='nose_border_polygon',
        left_hand_H='left_hand_border_polygon',
        right_hand_H='right_hand_border_polygon',
        left_palm_H='left_palm_border_polygon',
        right_palm_H='right_palm_border_polygon',
        left_hand_P='pose_left_hand_polygon',
        right_hand_P='pose_right_hand_polygon'
    )
    regions_dict = {k: keypoints_dict[v] for k, v in mapping.items()}
    return regions_dict


def get_regional_temperatures(tempC_array, keypoints_ir_dict):
    """
    Get temperature values within body regions of interest.
    """
    regions_dict = prepare_regions_dictionary(keypoints_dict=keypoints_ir_dict)
    # No need to resize tempC_array because the polygons use normalized coordinates
    regional_temperatures = \
        extract_raster_array_values_inside_multiple_polygons(
            raster=tempC_array,
            polygon_dict=regions_dict
        )
    return regional_temperatures


def compute_temperature_statistics(values, reference_temperature_offset=None):
    """
    Compute statistics (hot index, cold index, min, percentile25, percentile 75, max)
    for a numpy array of temperature values.
    """
    empty_value = np.nan
    if values is None or len(values) == 0:
        hot_index = empty_value
        cold_index = empty_value
        minimum = empty_value
        percentile_25 = empty_value
        percentile_75 = empty_value
        maximum = empty_value
    else:
        if reference_temperature_offset is None:
            print('compute_temperature_statistics(): No reference temperature offset supplied')
            return None
        values = values - reference_temperature_offset
        descending = sorted(values, reverse=True)
        # This approach to calculating hot index assumes that the temperature raster from which regions were extracted
        # has its initial (unscaled) resolution of 160 pixels W by 120 pixels H.
        if len(descending) >= 10:
            ten_highest = descending[0:10]
        else:
            ten_highest = descending
        hot_index = np.median(ten_highest)
        cold_index = np.median(values)
        minimum = np.min(values)
        percentile_25 = np.percentile(values, q=25)
        percentile_75 = np.percentile(values, q=75)
        maximum = np.max(values)

    stats = dict(
        HI=hot_index,
        CI=cold_index,
        min=minimum,
        p25=percentile_25,
        p75=percentile_75,
        max=maximum
    )
    return stats


def get_regional_temperature_statistics(tempC_array, ir_keypoints_dict, reference_temperature_offset=None):
    """
    Return statistics about temperatures in body regions.
    """
    regional_temperatures = \
        get_regional_temperatures(
            tempC_array=tempC_array,
            keypoints_ir_dict=ir_keypoints_dict
        )
    regional_temperature_stats = {
        k: compute_temperature_statistics(values=v, reference_temperature_offset=reference_temperature_offset) \
        for k, v in regional_temperatures.items()
    }
    return regional_temperature_stats


def round_dictionary_values(d, digits=2):
    """
    Round dictionary values to a specified number of digits after the decimal place.
    """
    rounded_dict = {k: (v if v is None else round(v, digits)) for k, v in d.items()}
    return rounded_dict


def flatten_regional_stats(regional_stats, digits=None):
    """
    Flatten dictionary of regional temperature statistics to convert from {{regionA:{stat1:value1, ..., }, ...}
    to  {regionA_stat1: value1, ...}.
    """
    stats_flattened = {
        f'{k_region}_{k_stat}': regional_stats[k_region][k_stat] \
        for k_region in regional_stats \
        for k_stat in regional_stats[k_region]
    }
    if digits is not None:
        result = round_dictionary_values(d=stats_flattened, digits=digits)
    else:
        result = stats_flattened
    return result
