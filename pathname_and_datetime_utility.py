import glob
import os
import re
from datetime import datetime

import pandas as pd

from config import FILENAME_DATETIME_FORMAT_GLOBAL


def get_temperature_pickle_pathname_from_vis_image_pathname(vis_image_pathname):
    """
    Find the pathname of the temperature raster pickle file corresponding
    to a visible (color) image file.
    """
    vis_image_dir, vis_image_filename = os.path.split(vis_image_pathname)
    datetime_string = extract_datetime_string(s=vis_image_filename)
    if datetime_string is None:
        return None
    temperature_pickle_dir = vis_image_dir.replace('Visual_Images', 'Pickle_Files')
    temperature_pickle_pathname = os.path.join(temperature_pickle_dir, f'temp_info_{datetime_string}.pickle')
    if not os.path.exists(temperature_pickle_pathname):
        return None
    return temperature_pickle_pathname


def extract_datetime_string(s):
    """
    Extract datetime string of form '20YY-mm-dd_HH-MM-SS'.
    """
    search_result = re.search(pattern=r'20\d\d\-\d+\-\d+\_\d+\-\d+\-\d+', string=s)
    if search_result is None:
        return None
    datetime_string = search_result[0]
    return datetime_string


def get_datetime_string_from_pathname(pathname):
    """
    Extract datetime string from filename at end of pathname
    """
    filename = os.path.split(pathname)[1]
    datetime_string = extract_datetime_string(s=filename)
    return datetime_string


def get_datetime_object_from_pathname(pathname, datetime_format=FILENAME_DATETIME_FORMAT_GLOBAL):
    """
    Extract datetime object from filename at end of pathname.
    """
    datetime_string = get_datetime_string_from_pathname(pathname=pathname)
    if datetime_string is None:
        return None
    datetime_object = datetime.strptime(datetime_string, datetime_format)
    return datetime_object


def my_strptime(s, format1, format2=None):
    """
    This version of strptime handles inconsistent datetime formatting
    in the files recording passive temperature reference sensor temperatures.
    """

    # If the datetime string include a decimal, assume that the seconds field has
    # been reported as a decimal and round the result to the nearest second.
    # This workaround is required because datetime.strptime() does not handle fractional seconds.
    # 2022-06-07: Turns out that datetime.strptime() can handle fractional seconds, per
    # https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior .
    # Can update this routine later.

    s_split = s.split('.')
    s1 = s_split[0]
    n = len(s_split)
    if n == 2:
        s2 = f'0.{s_split[1]}'
        extra_sec = round(float(s2))
    else:
        extra_sec = 0
    # Try datetime format format1, then format2, before giving up.
    try:
        datetime_object = datetime.strptime(s1, format1)
    except ValueError:
        if format2 is None:
            datetime_object = None
        else:
            try:
                datetime_object = datetime.strptime(s1, format2)
            except ValueError:
                datetime_object = None
    if datetime_object is not None:
        datetime_object += pd.Timedelta(seconds=extra_sec)
    return datetime_object

def fix_malformed_datetime_substring(s, target_dt_format='%Y-%m-%d_%H-%M-%S'):
    s2 = s
    for separator in ['_', '-']:
        if separator == '_':
            pattern = r'20\d\d\-\d+\-\d+_\d+\-\d+\-\d+'
            original_dt_format = '%Y-%m-%d_%H-%M-%S'
        elif separator == '-':
            pattern = r'20\d\d\-\d+\-\d+\-\d+\-\d+\-\d+'
            original_dt_format = '%Y-%m-%d-%H-%M-%S'
        m = re.search(pattern, s2)
        if m is not None:
            original_dt_string = m.group(0)
            try:
                dt = datetime.strptime(original_dt_string, original_dt_format)
                revised_dt_string = dt.strftime(target_dt_format)
                s2 = s2.replace(original_dt_string, revised_dt_string)
            except:
                pass
    return s2

def correct_datetimes_formats_in_filenames_and_foldernames(
        folder_pathname,
        start_date=None,
        end_date=None,
        implement=False,
        deep=False
):
    initial_directory = os.getcwd()
    files_renamed = 0
    os.chdir(folder_pathname)
    names = sorted(glob.glob(pathname='*'))
    n = len(names)
    print('Entered', folder_pathname, '(contains', n, 'files or folders)')
    for original in names:
        target_name = original
        revised = fix_malformed_datetime_substring(original)
        datetime_object = get_datetime_object_from_pathname(pathname=revised)
        skip = False
        if datetime_object is not None:
            if start_date is not None:
                skip |= datetime_object < start_date
            if end_date is not None:
                skip |= datetime_object > end_date
        if skip:
            continue
        different = revised != original
        if different:
            if implement:
                os.rename(original, revised)
                target_name = revised
                print(f'Renamed {original} → {revised}')
                files_renamed += 1
            else:
                print(f'Proposed change: {original} → {revised}')
        if deep:
            target_pathname = os.path.join(folder_pathname, target_name)
            if os.path.isdir(target_pathname):
                files_renamed += \
                    correct_datetimes_formats_in_filenames_and_foldernames(
                        folder_pathname=target_pathname,
                        implement=implement,
                        deep=deep,
                        start_date=start_date,
                        end_date=end_date
                    )
    os.chdir(initial_directory)
    return files_renamed



