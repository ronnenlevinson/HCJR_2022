import os
import pickle
import time

import cv2
import numpy as np

from graphics_utility import rescale_raster_values, resize_image
from pathname_and_datetime_utility import extract_datetime_string

# Hui: "The two human skin temperature ranges that you can use to filter your data process:
# 25 - 38C, and 22 - 38C"

SKIN_TEMPERATURE_LOWER_BOUND_C = 22  # °C
SKIN_TEMPERATURE_UPPER_BOUND_C = 38  # °C

# Zero degrees Celsius in Kelvin

ZERO_C_IN_K = 273.15


def convert_Kelvin_to_Celsius(T_K):
    """
    Convert temperature or temperature array from Kelvin to Celsius.
    """
    return T_K - ZERO_C_IN_K


def convert_Celsius_to_Kelvin(T_C):
    """
    Convert temperature or temperature array from Celsius to Kelvin.
    """
    return T_C + ZERO_C_IN_K


def read_tempC_array(
        pickle_pathname,
        scale=1,
        max_attempts=100,  # number of times to try reading the temperature raster from the pickle file
        delay_sec=0.1
):
    """
    Read a temperature raster (numpy array of 16-bit unsigned integers) from a pickle file,
    resize it according the supplied scale factor, and convert the temperature values from 100 * Kelvin to °C.
    For example, a raster value of 30,000 is 30,000 / 100 = 300 K = 26.85 °C.
    """

    # The function makes multiple attempts to load the pickle file because I was experiencing timeouts related
    # to Google Drive. This may no longer be an issue.
    file_loaded = False
    attempt = 0
    while attempt < max_attempts and not file_loaded:
        attempt += 1
        try:
            with open(pickle_pathname, "rb") as f:
                measurements = pickle.load(f)
            file_loaded = True
        except Exception as e:
            print('X', end='')
            time.sleep(delay_sec)
    if file_loaded:
        absolute_temp_times_100_original = measurements['ir_temp_array']  # 100 * absolute temperature [K]
        if scale == 1:
            absolute_temp_times_100 = absolute_temp_times_100_original
        else:
            # resize_image() works on any integer array, not just on images.
            absolute_temp_times_100 = resize_image(image=absolute_temp_times_100_original, scale=scale)
        tempC_array = convert_Kelvin_to_Celsius(absolute_temp_times_100 / 100.0)
        if attempt == 1:
            print('.', end='')  # Success after first attempt
        else:
            print('*', end='')  # Will retry
    else:
        tempC_array = None
        print('!', end='')  # Failed
    return tempC_array

def create_IR_image_from_temperature_pickle_file(
        pickle_pathname,
        verbose=False,
        write_IR_image_file=True,
        **kwargs
):
    """
    Generate an IR (thermal) image from a pickled temperature raster, and write it to file (optional).
    """
    pickle_dir, pickle_file = os.path.split(pickle_pathname)
    datetime_string = extract_datetime_string(pickle_file)
    if datetime_string is None:
        return None
    tempC_array_unbounded = read_tempC_array(pickle_pathname=pickle_pathname)
    if tempC_array_unbounded is None:
        return None
    # The bounded temperature raster is used only to generate the IR image.
    tempC_array_bounded = bound_numpy_array(tempC_array_unbounded, **kwargs)
    image_array = \
        rescale_raster_values(
            raster=tempC_array_bounded,
            new_min=0,
            new_max=255,
            new_type=np.uint8
        )
    # We use folder and file names beginning 'IR2' to distinguish these regenerated IR images
    # from those collected during the trials.
    ir2_image_dir = pickle_dir.replace('Pickle_Files', 'IR2_Images')
    ir2_image_file = f'ir2_{datetime_string}.png'
    if not os.path.exists(ir2_image_dir):
        os.mkdir(ir2_image_dir)
    ir2_image_path = os.path.join(ir2_image_dir, ir2_image_file)
    if write_IR_image_file:
        cv2.imwrite(filename=ir2_image_path, img=image_array)
        if verbose:
            print(f'Wrote {ir2_image_path}')
    ir_image_data = dict(ir_image_pathname=ir2_image_path, tempC_array_unbounded=tempC_array_unbounded,
                         image_array=image_array)
    return ir_image_data


def create_sharpened_IR_image_from_pickle_file(
        pickle_pathname,
        temperature_span=None,
        **kwargs
):
    """
    Variation on IR-image generation that limits the temperature values to minimum value + temperature_span.
    This is used in the routine that lets the operator locate a passive or active temperature reference in
    an IR (thermal) image.
    """
    tempC_array = read_tempC_array(pickle_pathname=pickle_pathname, **kwargs)
    if tempC_array is None:
        return None
    if temperature_span == None:
        tempC_array_bounded = tempC_array
    else:
        lower_bound = np.min(tempC_array)
        upper_bound = lower_bound + temperature_span
        tempC_array_bounded = bound_numpy_array(tempC_array, lower_bound=lower_bound, upper_bound=upper_bound)
    image = \
        rescale_raster_values(
            raster=tempC_array_bounded,
            new_min=0,
            new_max=255,
            new_type=np.uint8
        )
    result = dict(image=image, lower_bound=lower_bound, upper_bound=upper_bound, tempC_array=tempC_array)
    return result


def bound_numpy_array(
        numpy_array,
        lower_bound=None,
        upper_bound=None
):
    """
    Apply to a numpy array a lower-bound (x[x<lower_bound] = lower_bound)
    and/or an upper_bound (x[x>upper_bound] = upper_bound) if specified.
    """
    numpy_array_bounded = numpy_array.copy()
    if lower_bound is not None:
        numpy_array_bounded[numpy_array < lower_bound] = lower_bound
    if upper_bound is not None:
        numpy_array_bounded[numpy_array > upper_bound] = upper_bound
    return numpy_array_bounded


