import glob
import os
import pickle
import time

import numpy as np
import pandas as pd

from config import REGENERATE_GLOBALS, EXTRA_SCENE_DATA_DIR_GLOBAL, PTR_CONTACT_TEMPERATURE_DIR_GLOBAL, \
    CBE_LAPTOP_DIR_GLOBAL, THERMAL_SENSATION_VOTE_DIR_GLOBAL, ATR_TEMPERATURE_C

from pathname_and_datetime_utility import get_temperature_pickle_pathname_from_vis_image_pathname, get_datetime_object_from_pathname, \
    my_strptime

from general_utility import report_elapsed_time


def read_all_passive_reference_temperature_data_files(regenerate=REGENERATE_GLOBALS):
    """
    Read all passive temperature reference (PTR) contact temperature data files collected over the 2021 trials
    and prepare a harmonized time series (data frame) of PTR contact temperatures.
    """
    combined_passive_reference_temperature_data_pathname = \
        os.path.join(EXTRA_SCENE_DATA_DIR_GLOBAL, 'combined_passive_reference_temperature_data.pickle')
    if os.path.exists(combined_passive_reference_temperature_data_pathname) and not regenerate:
        with open(combined_passive_reference_temperature_data_pathname, 'rb') as f:
            result = pickle.load(file=f)
    else:
        pathnames = sorted(glob.glob(os.path.join(PTR_CONTACT_TEMPERATURE_DIR_GLOBAL, '*Temp_data.csv')))
        df_list = []
        for pathname in pathnames:
            df = pd.read_csv(pathname)
            fieldnames = df.columns
            if 'Emmitter_Temperature' in fieldnames:
                temperature_field = 'Emmitter_Temperature'
            elif 'Emitter_Temperature' in fieldnames:
                temperature_field = 'Emitter_Temperature'
            elif 'Emitter_Temperature_Sensor' in fieldnames:
                temperature_field = 'Emitter_Temperature_Sensor'
            elif 'Reference_Temperature_Sensor' in fieldnames:
                temperature_field = 'Reference_Temperature_Sensor'
            else:
                temperature_field = None
            if temperature_field is not None:
                datetime_objects = [my_strptime(s, format1='%Y-%m-%d-%H-%M-%S', format2='%Y-%m-%d %H:%M:%S') for s in
                                    df['Timestamp']]
                if None not in datetime_objects:
                    reference_temperatures = df[temperature_field]
                    df1 = pd.DataFrame(
                        dict(datetime=datetime_objects, reference_temperature=reference_temperatures)).set_index(
                        'datetime')
                    df_list.append(df1)
        df_all0 = pd.concat(df_list)
        df_all0.sort_index(inplace=True)
        df_all1 = df_all0[~df_all0.index.duplicated(keep='first')]
        df_all2 = df_all1.resample('1S').ffill(limit=3)
        df_all3 = df_all2.dropna()
        result = df_all3
        with open(combined_passive_reference_temperature_data_pathname, 'wb') as f:
            pickle.dump(result, file=f)
        print('Wrote ', combined_passive_reference_temperature_data_pathname)
    return result


def get_trial_location(scene_datetime_object):
    """
    Return the trial location based on scene datetime.
    """
    if scene_datetime_object < pd.Timestamp('2021-07-07'):
        location = 'carport'
    elif scene_datetime_object < pd.Timestamp('2021-09-21'):
        location = 'CBE'
    elif scene_datetime_object < pd.Timestamp('2021-12-03'):
        location = 'DSV'
    elif scene_datetime_object < pd.Timestamp('2022-06-01'):
        location = "Alex's home"
    else:
        location = 'CBE'
    return location


def passive_reference_temperature_available_for_scene(scene_datetime_object):
    """
    Return the availability of a contact temperature from a passive temperature reference (PTR)
    in the scene.
    """
    available = scene_datetime_object in scene_passive_reference_temperatures_global.index
    return available


def active_reference_temperature_available_for_scene(scene_datetime_object):
    """
    Return the availability of a temperature setpoint from an active temperature reference (ATR)
    in the scene.
    """
    available = get_trial_location(scene_datetime_object) == 'CBE'
    return available


def get_vis_image_pathnames_of_all_usable_scenes(regenerate=REGENERATE_GLOBALS):
    """
    Return numpy array of pathnames of all usable scenes,
    meaning those with both a visible (color) image and a temperature raster.
    """
    all_usable_scenes_pathname = \
        os.path.join(EXTRA_SCENE_DATA_DIR_GLOBAL, 'all_usable_scenes.pickle')
    if os.path.exists(all_usable_scenes_pathname) and not regenerate:
        with open(all_usable_scenes_pathname, 'rb') as f:
            df_scenes_usable = pickle.load(file=f)
    else:
        vis_image_pathnames0 = \
            np.array(sorted(glob.glob(os.path.join(CBE_LAPTOP_DIR_GLOBAL, '*', 'Visual_Images/vis_*.jpg'))))
        scene_datetime_objects0 = np.array([get_datetime_object_from_pathname(pathname) for pathname in vis_image_pathnames0])
        df_scenes = \
            pd.DataFrame(
                dict(scene_datetime_object=scene_datetime_objects0, vis_image_pathname=vis_image_pathnames0)
            )# .set_index('scene_datetime_object')
        df_scenes.dropna(subset=['scene_datetime_object'], inplace=True)
        df_scenes.drop_duplicates(subset=['scene_datetime_object'], inplace=True)
        df_scenes.set_index('scene_datetime_object', inplace=True)
        # index0 = df_scenes.index
        # index1 = index0.dropna()
        # index2 = index1.drop_duplicates()
        # df_scenes = df_scenes.loc[index2, :]
        vis_image_pathnames = df_scenes['vis_image_pathname']
        print('Number of vis image files = ', len(vis_image_pathnames))
        # The routine to get the temperature pickle pathname will eliminate those visible image
        # and temperature raster files with improperly formatted datetime strings
        # (e.g., of the form YYYY-MM-DD.dayfraction).

        temperature_pickle_pathnames = \
            np.array([get_temperature_pickle_pathname_from_vis_image_pathname(vis_image_pathname) \
                      for vis_image_pathname in vis_image_pathnames])
        df_scenes['temperature_pickle_pathname'] = temperature_pickle_pathnames
        # selector2 = pd.notnull(temperature_pickle_pathnames)
        df_scenes_usable = df_scenes.dropna(subset=['temperature_pickle_pathname'])
        with open(all_usable_scenes_pathname, 'wb') as f:
            pickle.dump(df_scenes_usable, file=f)
    return df_scenes_usable





def join_vote_files(regenerate=REGENERATE_GLOBALS):
    """
    Compile two dataframes: one containing all thermal sensation votes reported in the carport, chamber, and DSV trials,
    and the second omitting those rows in which there was a comfort vote but no sensation vote. Write the former to Excel
    if regenerating the dataframes.
    """
    votes_joined_pathname = os.path.join(EXTRA_SCENE_DATA_DIR_GLOBAL, 'All HCJR votes.xlsx')
    if os.path.exists(votes_joined_pathname) and not regenerate:
        df_joined = pd.read_excel(votes_joined_pathname, index_col='DATETIME', parse_dates=True)
    else:
        vote_files = glob.glob(os.path.join(THERMAL_SENSATION_VOTE_DIR_GLOBAL, "[0-9]_*.xlsx"))
        dfs = [pd.read_excel(vote_file, index_col='DATETIME', parse_dates=True) for vote_file in vote_files]
        df_joined = dfs[0].join(dfs[1:], how='outer', sort=True)
        df_joined.index = pd.to_datetime(df_joined.index)
        carport_vote_is_sensation = df_joined['CARPORT_survey2'] == 1
        carport_vote = df_joined['CARPORT_vote']
        carport_simplified_sensation_vote = \
            np.where(
                carport_vote_is_sensation,
                1 * (carport_vote > 1) + (-1) * (carport_vote < -1),
                np.nan
            )
        chamber_2021_sensation_vote = df_joined['CHAMBER_2021_Sensation_Number']
        chamber_2021_simplified_sensation_vote = \
            np.where(
                pd.isna(chamber_2021_sensation_vote),
                np.nan,
                1 * (chamber_2021_sensation_vote >= 1) + (-1) * (chamber_2021_sensation_vote <= -1)
            )
        chamber_2022_sensation_vote = df_joined['CHAMBER_2022_Thermal sensation vote (ASHRAE 7-point scale)']
        chamber_2022_simplified_sensation_vote = \
            np.where(
                pd.isna(chamber_2022_sensation_vote),
                np.nan,
                1 * (chamber_2022_sensation_vote >= 1) + (-1) * (chamber_2022_sensation_vote <= -1)
            )

        DSV_sensation_vote = df_joined['DSV_Sensation ("Please rate your thermal sensation right now")']
        DSV_simplified_sensation_vote = \
            np.where(
                pd.isna(DSV_sensation_vote),
                np.nan,
                1 * ((DSV_sensation_vote == 'Warm') | (DSV_sensation_vote == 'Hot')) + \
                (-1) * ((DSV_sensation_vote == 'Cool') | (DSV_sensation_vote == 'Cold'))
            )

        simplified_sensation_vote = \
            np.where(
                np.isfinite(carport_simplified_sensation_vote),
                carport_simplified_sensation_vote,
                np.where(
                    np.isfinite(chamber_2021_simplified_sensation_vote),
                    chamber_2021_simplified_sensation_vote,
                    np.where(
                        np.isfinite(chamber_2022_simplified_sensation_vote),
                        chamber_2022_simplified_sensation_vote,
                        DSV_simplified_sensation_vote
                    )
                )
            )

        sweating_vote = df_joined['CHAMBER_2022_Sweating status (1=sweating; 0.5=slightly sweating; 0=not sweating)']
        df_joined['CARPORT_simplified_sensation_vote'] = carport_simplified_sensation_vote
        df_joined['CHAMBER_simplified_2021_sensation_vote'] = chamber_2021_sensation_vote
        df_joined['CHAMBER_simplified_2022_sensation_vote'] = chamber_2022_sensation_vote
        df_joined['DSV_simplified_sensation_vote'] = DSV_sensation_vote
        df_joined['SIMPLIFIED_sensation_vote'] = simplified_sensation_vote
        df_joined['sweating_vote'] = sweating_vote
        df_joined.to_excel(votes_joined_pathname, na_rep='#N/A')
        print(f'Wrote {votes_joined_pathname}')
    df_joined_useful = df_joined.loc[np.isfinite(df_joined['SIMPLIFIED_sensation_vote']), :]
    return df_joined, df_joined_useful




def find_vote_datetime_object_nearest_scene(scene_datetime_object):
    """
    Find the datetime of the thermal sensation vote nearest that of a given scene.
    """
    vote_datetime_objects = all_useful_HCJR_votes_global.index
    abs_time_differences = abs(vote_datetime_objects - scene_datetime_object)
    ser = pd.Series(abs_time_differences, index=vote_datetime_objects)
    nearest_vote_datetime_object = ser.idxmin()
    return nearest_vote_datetime_object


def match_scenes_to_votes(regenerate=REGENERATE_GLOBALS):
    """
    Return a dataframe with the datetime object of the thermal sensation vote nearest each usable scene.
    The index of the dataframe is the datetime object of the scene.
    """
    vote_nearest_scene_pathname = os.path.join(EXTRA_SCENE_DATA_DIR_GLOBAL, "Vote nearest scene.csv")
    if os.path.exists(vote_nearest_scene_pathname) and not regenerate:
        df_vote_nearest_scene = pd.read_csv(vote_nearest_scene_pathname, index_col=0, parse_dates=[0, 1])
    else:
        # Apply np.unique() in case there is more than one scene with the same datetime.
        # np.unique() defaults to keeping only the first instance of each duplicated element.
        scene_datetime_objects = df_all_usable_scenes_global.index # [get_datetime_object_from_pathname(pathname) for pathname in all_usable_scenes_global]
        nearest_vote_datetime_objects = \
            [find_vote_datetime_object_nearest_scene(scene_datetime_object=scene_datetime_object) \
             for scene_datetime_object in scene_datetime_objects]
        ser = pd.Series(nearest_vote_datetime_objects, index=scene_datetime_objects)
        df_vote_nearest_scene = pd.DataFrame(dict(vote_nearest_scene=ser))
        df_vote_nearest_scene.to_csv(vote_nearest_scene_pathname)
        print('Wrote ', vote_nearest_scene_pathname)
    return df_vote_nearest_scene


def get_datetime_objects_of_scenes_near_votes(window_in_sec):
    """
    Return the datetime objects of those scenes within a given window (timedelta)
    of a thermal sensation vote. For example, a time window of 10 seconds
    would select scenes within plus or minus 10 seconds of a vote.
    """
    df = df_vote_nearest_scene_global.copy()
    time_difference = df['vote_nearest_scene'] - df.index
    if type(window_in_sec) is list:
        time_delta = np.array([pd.Timedelta(seconds=seconds) for seconds in window_in_sec])
    else:
        time_delta = pd.Timedelta(seconds=window_in_sec)
    selector = abs(time_difference) <= time_delta
    datetimes_of_scenes_near_votes = df.loc[selector, :]
    return datetimes_of_scenes_near_votes


def tabulate_usable_scenes_and_reference_temperatures_near_votes(window_in_sec=60, regenerate=REGENERATE_GLOBALS):
    """
    Create for all usable scenes within some window (timedelta) of a thermal sensation vote
    a dataframe with visible image pathname, temperature raster pickle file pathname,
    datetime object of the nearest vote, PTR contact temperature (NaN if unavailable),
    and ATR contact temperature (NaN if unavailable).
    """
    usable_scenes_and_reference_temperatures_near_votes_pathname = \
        os.path.join(EXTRA_SCENE_DATA_DIR_GLOBAL, 'usable_scenes_and_reference_temperatures_near_votes.pickle')
    if os.path.exists(usable_scenes_and_reference_temperatures_near_votes_pathname) and not regenerate:
        with open(usable_scenes_and_reference_temperatures_near_votes_pathname, 'rb') as f:
            usable_scenes_and_reference_temperatures_near_votes = pickle.load(file=f)
    else:
        scene_datetime_objects = df_vote_nearest_scene_global.index
        nearest_vote_datetime_objects = df_vote_nearest_scene_global['vote_nearest_scene']
        nearest_simplified_sensation_votes = \
            [all_useful_HCJR_votes_global.loc[nearest_vote_datetime_object, 'SIMPLIFIED_sensation_vote'] \
             for nearest_vote_datetime_object in nearest_vote_datetime_objects]
        half_hour_in_sec = 30 * 60 # seconds
        adaptive_time_window = \
            [window_in_sec if nearest_simplified_sensation_vote < 1 else half_hour_in_sec \
             for nearest_simplified_sensation_vote in nearest_simplified_sensation_votes]

        datetimes_of_scenes_near_votes = get_datetime_objects_of_scenes_near_votes(window_in_sec=adaptive_time_window)
        selector1 = [scene_datetime_object in datetimes_of_scenes_near_votes.index \
                     for scene_datetime_object in scene_datetime_objects]
        scene_datetime_objects1 = scene_datetime_objects[selector1]
        vis_image_pathnames1 = df_all_usable_scenes_global.loc[selector1, 'vis_image_pathname']
        temperature_pickle_pathnames1 = df_all_usable_scenes_global.loc[selector1, 'temperature_pickle_pathname']
        nearest_vote_datetime_objects1 = df_vote_nearest_scene_global.loc[scene_datetime_objects1, 'vote_nearest_scene']
        nearest_simplified_sensation_votes1 = \
            [all_useful_HCJR_votes_global.loc[nearest_vote_datetime_object, 'SIMPLIFIED_sensation_vote'] \
             for nearest_vote_datetime_object in nearest_vote_datetime_objects1]
        nearest_sweating_votes1 = \
            [all_useful_HCJR_votes_global.loc[nearest_vote_datetime_object, 'sweating_vote'] \
             for nearest_vote_datetime_object in nearest_vote_datetime_objects1]
        PTR_contact_temperatures = [
            scene_passive_reference_temperatures_global.loc[scene_datetime_object, 'reference_temperature'] \
                if passive_reference_temperature_available_for_scene(scene_datetime_object) else np.nan \
            for scene_datetime_object in scene_datetime_objects1
        ]
        ATR_contact_temperatures = [
            ATR_TEMPERATURE_C \
                if active_reference_temperature_available_for_scene(scene_datetime_object) else np.nan \
            for scene_datetime_object in scene_datetime_objects1
        ]
        df = pd.DataFrame(
            dict(
                scene_datetime_object=scene_datetime_objects1,
                nearest_vote_datetime_object=nearest_vote_datetime_objects1,
                nearest_simplified_sensation_vote=nearest_simplified_sensation_votes1,
                nearest_sweating_vote=nearest_sweating_votes1,
                vis_image_pathname=vis_image_pathnames1,
                temperature_pickle_pathname=temperature_pickle_pathnames1,
                PTR_contact_temperature=PTR_contact_temperatures,
                ATR_contact_temperature=ATR_contact_temperatures
            )
        ).set_index('scene_datetime_object')
        # df.set_index('scene_datetime_object', inplace=True)
        usable_scenes_and_reference_temperatures_near_votes = df
        with open(usable_scenes_and_reference_temperatures_near_votes_pathname, 'wb') as f:
            pickle.dump(usable_scenes_and_reference_temperatures_near_votes, file=f)
        print('Wrote ', usable_scenes_and_reference_temperatures_near_votes_pathname)
    return usable_scenes_and_reference_temperatures_near_votes

if 'scene_passive_reference_temperatures_global' not in globals():
    print('Generating scene_passive_reference_temperatures_global')
    start_time1 = time.time()
    scene_passive_reference_temperatures_global = read_all_passive_reference_temperature_data_files()
    report_elapsed_time(start_time=start_time1)

if 'df_all_usable_scenes_global' not in globals():
    print('Generating all_usable_scenes_global')
    start_time2 = time.time()
    df_all_usable_scenes_global = get_vis_image_pathnames_of_all_usable_scenes()
    report_elapsed_time(start_time=start_time2)

if 'all_HCJR_votes_global' not in globals():
    print('Generating all_HCJR_votes_global')
    start_time3 = time.time()
    all_HCJR_votes_global, all_useful_HCJR_votes_global = join_vote_files()
    report_elapsed_time(start_time=start_time3)

if 'df_vote_nearest_scene_global' not in globals():
    print('Generating df_vote_nearest_scene_global')
    start_time4 = time.time()
    df_vote_nearest_scene_global = match_scenes_to_votes()
    report_elapsed_time(start_time=start_time4)

if 'usable_scenes_and_reference_temperatures_near_votes_global' not in globals():
    print('Generating usable_scenes_and_reference_temperatures_near_votes_global')
    start_time5 = time.time()
    usable_scenes_and_reference_temperatures_near_votes_global = tabulate_usable_scenes_and_reference_temperatures_near_votes()
    report_elapsed_time(start_time=start_time5)