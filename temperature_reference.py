import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from EasyROI import EasyROI

from generate_IR_image import create_sharpened_IR_image_from_pickle_file, convert_Celsius_to_Kelvin
from graphics_utility import extract_raster_values_between_two_nested_polygons, extract_raster_values_inside_polygon, \
    get_height_and_width, draw_outlined_text_on_image
from config import FILENAME_DATETIME_FORMAT_GLOBAL, IR_SCALE_GLOBAL, ESC_GLOBAL, SPACE_GLOBAL
from load_past_trial_data import usable_scenes_and_reference_temperatures_near_votes_global
from ROI_utility import load_roi_dict, draw_roi_polygons_on_image, get_ATR_border, get_PTR_borders, \
    get_PTR_and_ATR_borders, save_roi_dict

# PTR = passive temperature reference (floating temperature is measured)
# ATR = active temperature reference (temperature is regulated)

# The PTR is an aluminum plate (TE about 0.10; check w/Howdy) with a painted central square
# that has high thermal emittance (TE about 0.90) and an outer region surfaced with crinkled aluminized mylar
# (check TE). The LBNL ATR used in climate-chamber trials has TE 0.95 and was held at 35 °C.

def get_PTR_temperatures(tempC_array, inner_PTR_border, outer_PTR_border):
    """
    Return 1-D arrays of the radiometric temperature values within the aluminized mylar (low-TE) region
    and within the painted metal (high-TE) region of the PTR. Each PTR region is approximated as a gray body
    at the same temperature as its background.
    """
    aluminized_mylar_temperatures = extract_raster_values_between_two_nested_polygons(
        raster=tempC_array,
        inner_polygon=inner_PTR_border,
        outer_polygon=outer_PTR_border
    )
    painted_metal_temperatures = extract_raster_values_inside_polygon(
        raster=tempC_array,
        polygon=inner_PTR_border
    )
    result = dict(aluminized_mylar_C=aluminized_mylar_temperatures, painted_metal_C=painted_metal_temperatures)
    return result


def get_ATR_temperatures(tempC_array, ATR_border):
    """
    Return 1-D array of the radiometric temperature values within the heated region of the ATR. The ATR region
    (TE 0.95) is approximated as a black body.
    """
    ATR_temperatures = extract_raster_values_inside_polygon(
        raster=tempC_array,
        polygon=ATR_border
    )
    result = dict(ATR_C=ATR_temperatures)
    return result


def compute_mean_radiometric_temperatures_of_PTR_and_ATR(
        tempC_array,
        polygons
):
    """
    Return median radiometric temperatures of the painted metal and alumimized mylar
    regions of the PTR (if present) and of the heated region of the ATR (if present).
    Each PTR surface is approximated as a gray body at the same temperature as its background,
    which behaves like a black body. The ATR surface (TE 0.95) is approximated as a black body.
    """
    if polygons is None:
        return None
    median_temperatures = dict()
    # Can probably remove the try-except statement.
    try:
        if 'inner_PTR' in polygons and 'outer_PTR' in polygons:
            PTR_temperatures_C = get_PTR_temperatures(
                tempC_array=tempC_array,
                inner_PTR_border=polygons['inner_PTR'],
                outer_PTR_border=polygons['outer_PTR']
            )
            for region in ['painted_metal_C', 'aluminized_mylar_C']:
                median_temperatures[region] = np.median(PTR_temperatures_C[region])
        if 'ATR' in polygons:
            ATR_temperatures_C = get_ATR_temperatures(tempC_array, ATR_border=polygons['ATR'])
            median_temperatures['ATR_C'] = np.median(ATR_temperatures_C['ATR_C'])
    except Exception as error:
        print('Error in compute_mean_radiometric_temperatures_of_PTR_and_ATR: ', error)
        print('tempC_array = ', tempC_array)
        print('polygons = ', polygons)
        median_temperatures = None
    return median_temperatures


def compute_apparent_PTR_temperature_and_background_temperature(
        tempC_array,
        inner_PTR_border,
        outer_PTR_border,
        aluminized_mylar_TE=0.05,
        painted_metal_TE=0.90
):
    """
    Compute background-corrected "apparent" temperatures within each of the PTR regions. This routine is NOT IN USE
    because (a) it's unclear whether the specular low-TE region of the PTR (aluminized mylar)
    sees the same background radiation as the diffuse high-TE region of the PTR (white-painted metal);
    and (b) since the true temperature of the PTR is likely close to that of the background radiation
    if the room temperature is roughly uniform, it's simpler to treat the high-TE region of the PTR as
    a black body.
    """
    PTR_temperatures_C = get_PTR_temperatures(
        tempC_array=tempC_array,
        inner_PTR_border=inner_PTR_border,
        outer_PTR_border=outer_PTR_border
    )

    aluminized_mylar_temperature_C_median = np.median(PTR_temperatures_C['aluminized_mylar_C'])
    painted_metal_temperature_C_median = np.median(PTR_temperatures_C['painted_metal_C'])

    # Assume that the thermal camera has been programmed to assign a thermal emittance of 1 to its target.
    # Let
    #     radiometric target temperature T_r  = absolute target temperature reported by camera, raw
    #     apparent target temperature T_a = absolute target temperature reported by camera, subsequently corrected for reflection of background radiation
    #     background temperature T_b = absolute temperature of target’s enclosure, such as a room
    #     true target temperature T_t = absolute temperature of target determined with a contact thermometer (e.g., a thermistor)
    #
    # If the enclosure is much larger than the target, we can treat the enclosure as a black body (thermal emittance = 1).
    # Let surfaces L and H represent the low thermal emittance and high thermal emittance areas
    # of the passive temperature reference (PTR, surface P). Assume that T_L=T_H=T_P.
    #
    # The following calculations derive from solving the radiative energy balance at the PTR's surface.

    T_L_r = convert_Celsius_to_Kelvin(aluminized_mylar_temperature_C_median)
    T_H_r = convert_Celsius_to_Kelvin(painted_metal_temperature_C_median)
    epsilon_L = aluminized_mylar_TE
    epsilon_H = painted_metal_TE
    x = ((epsilon_H - 1) * T_L_r ** 4 - (epsilon_L - 1) * T_H_r ** 4) / (epsilon_H - epsilon_L)
    y = (epsilon_H * T_L_r ** 4 - epsilon_L * T_H_r ** 4) / (epsilon_H - epsilon_L)
    T_P_a = x ** (1 / 4)
    T_b = y ** (1 / 4)
    result = dict(apparent_PTR_K=T_P_a, background_K=T_b)
    return result


def compute_apparent_surface_temperature_values_in_region(
        tempC_array,
        border,
        TE,
        background_temperature_K
):
    """
    Compute background-corrected "apparent" temperatures within a surface given its thermal emittance and
    background temperature. This routine is NOT IN USE because it's simpler to treat human skin (TE = 0.95-0.98)
    as a black body.
    """
    surface_temperature_values_C = extract_raster_values_inside_polygon(
        raster=tempC_array,
        polygon=border
    )
    # S = surface
    T_S_r = convert_Celsius_to_Kelvin(surface_temperature_values_C)
    T_b = background_temperature_K
    epsilon_S = TE
    # The following calculation works because T_S_r and T_b are numpy arrays
    T_s_a = ((T_S_r ** 4 - (1 - epsilon_S) * T_b ** 4) / epsilon_S) ** (1 / 4)
    result = dict(apparent_surface_K=T_s_a)
    return result


def compute_apparent_ATR_temperature(
        tempC_array,
        border,
        background_temperature_K,
        TE=0.95
):
    """
    Compute median apparent temperature of ATR. This function is NOT IN USE because it's simpler to treat
    the heated region of the ATR (TE=0.95) as a black body.
    """
    apparent_ATR_temperature_values_K = \
        compute_apparent_surface_temperature_values_in_region(
            raster=tempC_array,
            polygon=border,
            TE=TE,
            background_temperature_K=background_temperature_K
        )
    T_ATR_K = np.median(apparent_ATR_temperature_values_K)
    result = dict(apparent_ATR_K=T_ATR_K)
    return result


def get_reference_temperatures_and_offset(scene_datetime_object, roi_medians):
    """
    Given the median radiometric temperatures of the high-TE (painted metal) and low-TE (aluminized mylar)
    regions of the PTR and/or the heated region of the ATR, look up the corresponding contact temperatures
    in the scene records and return the temperature offset = radiometric temperature - contact temperature
    for (a) the painted metal region of the PTR, if present in the scene; or (b) the heated region of the
    ATR, if present in the scene.
    """
    if scene_datetime_object is None or roi_medians is None or len(roi_medians) == 0:
        radiometric_temperature = np.nan
        contact_temperature = np.nan
    else:
        PTR_contact_temperature, ATR_contact_temperature = \
        usable_scenes_and_reference_temperatures_near_votes_global.loc[
            scene_datetime_object, ['PTR_contact_temperature', 'ATR_contact_temperature']]
        if 'painted_metal_C' in roi_medians:
            radiometric_temperature = roi_medians['painted_metal_C']
            contact_temperature = PTR_contact_temperature
        elif 'ATR_C' in roi_medians:
            radiometric_temperature = roi_medians['ATR_C']
            contact_temperature = ATR_contact_temperature
        else:
            radiometric_temperature = np.nan
            contact_temperature = np.nan
    temperature_offset = radiometric_temperature - contact_temperature
    result = dict(
        reference_radiometric_temperature=radiometric_temperature,
        reference_contact_temperature=contact_temperature,
        reference_temperature_offset=temperature_offset
    )
    return result


def summarize_reference_temperatures(scene_datetime_object, roi_medians):
    """
    Prepare a summary of reference temperatures to display onscreen.
    """
    if scene_datetime_object is None or roi_medians is None or len(roi_medians) == 0:
        return None
    roi_median_string = '\n'.join([f'{k}={v:.1f} C' for k, v in roi_medians.items()])
    s = roi_median_string
    PTR_contact_temperature, ATR_contact_temperature = \
        usable_scenes_and_reference_temperatures_near_votes_global.loc[
            scene_datetime_object, ['PTR_contact_temperature', 'ATR_contact_temperature']
        ]
    if 'painted_metal_C' in roi_medians:
        painted_metal_minus_PTR_contact = roi_medians['painted_metal_C'] - PTR_contact_temperature
        s = \
            s + \
            f'\nPTR contact temperature={PTR_contact_temperature:.1f} C' + \
            f'\nPTR radiometric - contact={painted_metal_minus_PTR_contact:.1f} C'
    if 'ATR_C' in roi_medians:
        ATR_radiometric_minus_ATR_contact = roi_medians['ATR_C'] - ATR_contact_temperature
        s = \
            s + \
            f'\nATR contact temperature={ATR_contact_temperature:.1f} C' + \
            f'\nATR radiometric - contact={ATR_radiometric_minus_ATR_contact:.1f} C'
    return s


def mark_temperature_references(
        scenes_to_process=usable_scenes_and_reference_temperatures_near_votes_global,
        temperature_span=10,  # K
        roi_dict=None,
        roi_display_delay_msec=2000
):
    """
    Let the user scan through collected scenes and draw polygons locating the inner and outer borders of the
    passive temperature reference (PTR) and/or the heated region of the active temperature reference.
    """

    if roi_dict is None:
        roi_dict = load_roi_dict()
        if roi_dict is None:
            roi_dict = dict()
    # print(roi_dict)
    roi_dict_original = roi_dict.copy()
    scene_datetime_objects = scenes_to_process.index
    done = False
    n = len(scenes_to_process)
    i = 0
    forward_increment = 1
    backward_increment = -forward_increment
    no_increment = 0
    increment = forward_increment
    previous_increment = increment
    vis_window_name = 'VIS image'
    ir_window_name = 'IR image'
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 1
    lineType = 2
    roi_helper = EasyROI(verbose=False)
    while not done:
        scene = scenes_to_process.iloc[i]
        vis_image_pathname, temperature_pickle_pathname = \
            scene.loc[['vis_image_pathname', 'temperature_pickle_pathname']]
        scene_datetime_object = scene.name
        scene_datetime_string = scene_datetime_object.strftime(FILENAME_DATETIME_FORMAT_GLOBAL)
        vis_filename = os.path.split(vis_image_pathname)[1]
        vis_image = cv2.imread(vis_image_pathname)
        vis_image_height, vis_image_width = get_height_and_width(image=vis_image)
        vis_lowerLeft = (5, round(0.99 * (vis_image_height - 1)))
        vis_text = f'Scene {i:06d}: {vis_filename}'
        draw_outlined_text_on_image(
            image=vis_image,
            text=vis_text,
            bottomLeftCornerOfText=vis_lowerLeft,
            font=font,
            fontScale=fontScale,
            thickness=thickness,
            lineType=lineType
        )
        ir_result = create_sharpened_IR_image_from_pickle_file(
            pickle_pathname=temperature_pickle_pathname,
            temperature_span=temperature_span,
            scale=IR_SCALE_GLOBAL
        )
        ir_image_grayscale, lower_bound, upper_bound, tempC_array = \
            ir_result['image'], \
            ir_result['lower_bound'], \
            ir_result['upper_bound'], \
            ir_result['tempC_array']
        ir_image = cv2.cvtColor(ir_image_grayscale, cv2.COLOR_GRAY2BGR)
        ir_image_clean = ir_image.copy()
        ir_image_height, ir_image_width = get_height_and_width(image=ir_image)
        ir_lowerLeft = [5, round(0.99 * (ir_image_height - 1))]
        ir_middleLeft = [5, round(0.50 * (ir_image_height - 1))]
        ir_upperLeft = [5, round(0.05 * (ir_image_height - 1))]
        ir_lowerRight = [round(0.50 * (ir_image_width - 1)), round(0.8 * (ir_image_height - 1))]

        ir_text = f'Temperature span = {temperature_span:.1f} K (bounds = {lower_bound:.1f} - {upper_bound:.1f} C)'
        draw_outlined_text_on_image(
            image=ir_image,
            text=ir_text,
            bottomLeftCornerOfText=ir_lowerLeft,
            font=font,
            fontScale=fontScale,
            thickness=thickness,
            lineType=lineType
        )
        instructions = '\n'.join([
            'COMMA/PERIOD/SPACE_GLOBAL : scroll back/scroll forward/pause',
            '</> : one scene backward/forward',
            'p/n : previous/next ROI(s) scene',
            '-/= : reduce/decrease temperature span',
            '1/2/3 : add 1/2/3 ROI(s) (1=ATR, 2=PTR, 3=PTR+ATR)',
            'e : add ROI endpoint; d : delete ROI(s); s : save ROI dictionary',
            '[/] : move ROI(s) one scene backward/forward',
            'ESC_GLOBAL/Q : exit with/without saving edits'
        ])
        draw_outlined_text_on_image(
            image=ir_image,
            text=instructions,
            bottomLeftCornerOfText=ir_upperLeft,
            font=font,
            fontScale=fontScale,
            thickness=thickness,
            lineType=lineType,
            line_spacing_pixels=20
        )
        roi_datetimes = \
            pd.Series(
                [datetime.strptime(datetime_string, FILENAME_DATETIME_FORMAT_GLOBAL) \
                 for datetime_string in roi_dict.keys()]
            )
        scene_has_ROIs = scene_datetime_string in roi_dict
        if scene_has_ROIs:
            if roi_dict[scene_datetime_string] is None:
                draw_outlined_text_on_image(
                    image=ir_image,
                    text='* ROI endpoint *',
                    bottomLeftCornerOfText=ir_middleLeft,
                    font=font,
                    fontScale=2 * fontScale,
                    thickness=thickness,
                    lineType=lineType
                )
                roi_polygons = None
            else:
                roi_polygons = roi_dict[scene_datetime_string]
                draw_roi_polygons_on_image(
                    image=ir_image,
                    polygons=roi_polygons,
                    thickness=2
                )
        else:
            earlier_datetimes = roi_datetimes[roi_datetimes < scene_datetime_object]
            if len(earlier_datetimes) > 0:
                previous_datetime = max(earlier_datetimes)
                previous_datetime_string = previous_datetime.strftime(FILENAME_DATETIME_FORMAT_GLOBAL)
                roi_polygons = roi_dict[previous_datetime_string]
                ir_image = draw_roi_polygons_on_image(image=ir_image, polygons=roi_polygons, thickness=1)
            else:
                roi_polygons = None
        if roi_polygons is not None:
            roi_medians = \
                compute_mean_radiometric_temperatures_of_PTR_and_ATR(
                    tempC_array=tempC_array,
                    polygons=roi_polygons
                )
            text = \
                summarize_reference_temperatures(
                    scene_datetime_object=scene_datetime_object,
                    roi_medians=roi_medians
                )

            draw_outlined_text_on_image(
                image=ir_image,
                text=text,
                bottomLeftCornerOfText=ir_lowerRight,
                font=font,
                fontScale=fontScale,
                thickness=thickness,
                lineType=lineType,
                line_spacing_pixels=20
            )

        mosaic = np.concatenate((vis_image, ir_image), axis=1)
        mosaic_window_name = 'VIS and IR images'
        cv2.imshow(mosaic_window_name, mosaic)
        if scene_has_ROIs:
            key = cv2.waitKey(roi_display_delay_msec)
        else:
            key = cv2.waitKey(1)  # 1 msec
        if key > 0:
            char = chr(key)
            if char in [ESC_GLOBAL, 'Q']:
                done = True
            elif char == ',':
                increment = backward_increment
            elif char == '.':
                increment = forward_increment
            elif char == SPACE_GLOBAL:
                if increment == no_increment:
                    increment = previous_increment
                else:
                    previous_increment = increment
                    increment = no_increment
            elif char == '<':
                increment = no_increment
                i -= 1
            elif char == '>':
                increment = no_increment
                i += 1
            elif char == '-':
                if temperature_span <= 1:
                    temperature_span -= 0.1
                else:
                    temperature_span = round(temperature_span) - 1
                if temperature_span < 0.1:
                    temperature_span = 0.1
            elif char == '=':
                if temperature_span < 1:
                    temperature_span += 0.1
                else:
                    temperature_span = round(temperature_span) + 1
            elif char == '1':
                increment = no_increment
                ATR_border = get_ATR_border(roi_helper=roi_helper, image=ir_image_clean)
                if ATR_border is not None:
                    roi_dict[scene_datetime_string] = ATR_border
            elif char == '2':
                increment = no_increment
                PTR_borders = get_PTR_borders(roi_helper=roi_helper, image=ir_image_clean)
                if PTR_borders is not None:
                    roi_dict[scene_datetime_string] = PTR_borders
            elif char == '3':
                PTR_and_ATR_borders = get_PTR_and_ATR_borders(roi_helper=roi_helper, image=ir_image_clean)
                if PTR_and_ATR_borders is not None:
                    roi_dict[scene_datetime_string] = PTR_and_ATR_borders
            elif char in ['n', 'p'] and len(roi_dict) > 0:
                if char == 'n':
                    later_datetimes = roi_datetimes[roi_datetimes > scene_datetime_object]
                    if len(later_datetimes) > 0:
                        next_datetime = min(later_datetimes)
                        i = scene_datetime_objects.get_loc(next_datetime)
                        increment = no_increment
                elif char == 'p':
                    earlier_datetimes = roi_datetimes[roi_datetimes < scene_datetime_object]
                    if len(earlier_datetimes) > 0:
                        previous_datetime = max(earlier_datetimes)
                        i = scene_datetime_objects.get_loc(previous_datetime)
                        increment = no_increment
            elif char in ['[', ']']:
                if scene_has_ROIs:
                    if char == ']' and (i < n - 1):
                        increment = forward_increment
                        new_i = i + increment
                    elif char == '[' and (i > 0):
                        increment = backward_increment
                        new_i = i + increment
                    else:
                        new_i = None
                    if new_i is not None:
                        new_scene = scenes_to_process.iloc[new_i]
                        new_scene_datetime_object = new_scene.name
                        new_scene_datetime_string = new_scene_datetime_object.strftime(FILENAME_DATETIME_FORMAT_GLOBAL)
                        roi_dict[new_scene_datetime_string] = roi_dict[scene_datetime_string].copy()
                        roi_dict.pop(scene_datetime_string)
            elif char == 'd':
                increment = no_increment
                if scene_datetime_string in roi_dict:
                    roi_dict.pop(scene_datetime_string)
            elif char == 'e':
                increment = no_increment
                roi_dict[scene_datetime_string] = None
            elif char == 's':
                save_roi_dict(roi_dict=roi_dict)
        else:
            char = ''
        i += increment
        if i < 0:
            i = 0
        elif i >= n:
            i = n - 1
    if char == 'Q':
        roi_dict = roi_dict_original
    elif char == ESC_GLOBAL:
        save_roi_dict(roi_dict=roi_dict)
    cv2.destroyAllWindows()
