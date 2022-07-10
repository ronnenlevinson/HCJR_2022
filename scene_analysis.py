import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from generate_IR_image import create_IR_image_from_temperature_pickle_file
from graphics_utility import resize_image, draw_filled_polygon_on_image, draw_polygon_on_image, draw_circles_on_image, \
    get_height_and_width, draw_outlined_text_on_image
from config import CBE_LAPTOP_DIR_GLOBAL, VIS_SCALE_GLOBAL, IR_SCALE_GLOBAL, VIDEO_DIR_GLOBAL, SPACE_GLOBAL, ESC_GLOBAL
from image_transformation import flatten_transformation_matrix_to_semicolon_separated_string, \
    transform_face_nose_eyeglasses_hand_palm_polygons, cull_ir_hand_and_palm_polygons, summarize_homographies
from mediapipe_landmark_detection import draw_styled_landmarks, get_feature_landmarks_and_polygons_from_image, \
    create_holistic_model
from load_past_trial_data import usable_scenes_and_reference_temperatures_near_votes_global
from ROI_utility import find_most_recent_roi_polygons, draw_roi_polygons_on_image, load_roi_dict
from regional_temperature_extraction import flatten_regional_stats, round_dictionary_values, \
    get_regional_temperature_statistics
from temperature_reference import compute_mean_radiometric_temperatures_of_PTR_and_ATR, \
    summarize_reference_temperatures, get_reference_temperatures_and_offset


def summarize_scene(
        scene_number,
        scene_datetime_object,
        nearest_vote_datetime_object,
        nearest_simplified_sensation_vote,
        nearest_sweating_vote,
        homographies,
        regional_temperature_stats,
        reference_temperature_stats,
        visibility_stats_ir,
        min_detection_confidence_vis,
        min_detection_confidence_ir,
        stats_digits=2
):
    """
    Return dictionary detailing all scene statistics.
    """
    stats_flattened = flatten_regional_stats(regional_stats=regional_temperature_stats, digits=stats_digits)
    condition_numbers = {f'transformation_condition_number_{k}': v['condition_number'] for k, v in homographies.items()}
    transformation_matrix_strings = {
        f'transformation_matrix_{k}':
        flatten_transformation_matrix_to_semicolon_separated_string(v['transformation_matrix']) \
        for k, v in homographies.items()
    }
    condition_numbers_rounded = round_dictionary_values(d=condition_numbers, digits=1)
    reference_temperature_stats_rounded = round_dictionary_values(d=reference_temperature_stats, digits=stats_digits)
    visibility_stats_ir_rounded = round_dictionary_values(d=visibility_stats_ir, digits=stats_digits)
    summary_dict = \
        dict(
            scene_number=scene_number,
            scene_datetime_object=scene_datetime_object,
            nearest_vote_datetime_object=nearest_vote_datetime_object,
            nearest_simplified_sensation_vote=nearest_simplified_sensation_vote,
            nearest_sweating_vote=nearest_sweating_vote,
            min_detection_confidence_vis=min_detection_confidence_vis,
            min_detection_confidence_ir=min_detection_confidence_ir
        )
    summary_dict.update(condition_numbers_rounded)
    summary_dict.update(transformation_matrix_strings)
    summary_dict.update(reference_temperature_stats_rounded)
    summary_dict.update(visibility_stats_ir_rounded)
    summary_dict.update(stats_flattened)
    return summary_dict


def summarize_key_temperature_stats(regional_temperature_stats, digits=None):
    """
    Return string summarizing key temperature statistics in scene.
    """
    stats_flattened = flatten_regional_stats(regional_stats=regional_temperature_stats, digits=digits)
    field_lists = [['face_F_HI', 'nose_F_CI'], ['left_hand_P_CI', 'right_hand_P_CI']]
    line_strings = \
        [
        (lambda field_list: ', '.join([f'{field}={stats_flattened[field]:.1f} C' for field in field_list]))(field_list) \
        for field_list in field_lists
        ]
    summary_string = '\n'.join(line_strings)
    return summary_string


def get_visibility_stats(keypoints):
    """
    Return dictionary detailing key feature visibility statistics.
    """
    PF_vis = keypoints['pose_face_visibilities']
    PLH_vis = keypoints['pose_left_hand_visibilities']
    PRH_vis = keypoints['pose_right_hand_visibilities']
    if PF_vis is None:
        pose_face_visibility_min = 0
    else:
        pose_face_visibility_min = min(PF_vis)
    if PLH_vis is None:
        pose_left_hand_visibility_min = 0
    else:
        pose_left_hand_visibility_min = min(PLH_vis)
    if PRH_vis is None:
        pose_right_hand_visibility_min = 0
    else:
        pose_right_hand_visibility_min = min(PRH_vis)

    stats = \
        dict(
            face_P_IR_visibility_min=pose_face_visibility_min,
            left_hand_P_IR_visibility_min=pose_left_hand_visibility_min,
            right_hand_P_IR_visibility_min=pose_right_hand_visibility_min
        )
    return stats


def summarize_visibilities(keypoints):
    """
    Generate string summarizing visibilities of key features.
    """
    stats = get_visibility_stats(keypoints)
    #     print('visibility stats=',stats)
    summary_string = f"face_P_visibility_min={stats['face_P_IR_visibility_min']:.2f}, left_hand_P_visibility_min={stats['left_hand_P_IR_visibility_min']:.2f}, right_hand_P_visibility_min={stats['right_hand_P_IR_visibility_min']:.2f}"
    return summary_string


def render_features_on_image(
    image,
    keypoints,
    image_pathname,
    draw_all_landmarks=False,
    show_vertices=False,
    show_filename=False,
    draw_eye_centers=False,
    draw_face_nose_and_hand_borders=True,
    draw_face_and_hand_pose_landmarks=False,
    image_label=None,
    fill_polygons=False,
    extra_text=None,
    scale_factor_for_display=1
):
    """
    Render MediaPipe-derived features, including face oval, nose border, eyeglasses border,
    hand/palm borders (hands model), hand borders (pose model), and/or pose landmarks, on image.
    """
    resized_image = resize_image(image=image, scale=scale_factor_for_display)
    if draw_all_landmarks:
        draw_styled_landmarks(resized_image, keypoints['results'])
    polygon_fill_color = dict()
    polygon_outline_color = dict()
    landmark_circle_color = dict()
    if fill_polygons:
        if draw_face_nose_and_hand_borders:
           polygon_fill_color |= \
                dict(
                    face_oval_polygon='green',
                    nose_border_polygon='cyan',
                    left_hand_border_polygon='magenta',
                    right_hand_border_polygon='green',
                    left_palm_border_polygon='blue',
                    right_palm_border_polygon='maroon'
                )
        if draw_face_and_hand_pose_landmarks:
            polygon_fill_color |= \
                dict(
                    pose_left_hand_polygon='cyan',
                    pose_right_hand_polygon='orange'
                )
    if draw_face_nose_and_hand_borders:
        polygon_outline_color |= \
            dict(
                face_oval_polygon='green',
                nose_border_polygon='cyan',
                eyeglasses_border_polygon='darkorange',
                left_eye_border_polygon='pink',
                right_eye_border_polygon='aqua',
                left_hand_border_polygon='magenta',
                right_hand_border_polygon='gold',
                left_palm_border_polygon='blue',
                right_palm_border_polygon='maroon'
            )
    if draw_face_and_hand_pose_landmarks:
        polygon_outline_color |= \
            dict(
                pose_left_hand_polygon='cyan',
                pose_right_hand_polygon='orange'
            )
        landmark_circle_color |= \
            dict(
                pose_face_landmark_points='red',
                pose_left_hand_landmark_points='green',
                pose_right_hand_landmark_points='blue'
             )
    if draw_eye_centers:
        landmark_circle_color |= dict(eye_centers='gold')
    for region, colorname in polygon_fill_color.items():
        draw_filled_polygon_on_image(
            image=resized_image,
            polygon=keypoints[region],
            colorname=colorname
        )
    for region, colorname in polygon_outline_color.items():
        draw_polygon_on_image(
            image=resized_image,
            polygon=keypoints[region],
            colorname=colorname,
            thickness=1,
            show_vertices=show_vertices
        )
    for region, colorname in landmark_circle_color.items():
        draw_circles_on_image(
            image=resized_image,
            centers=keypoints[region],
            radius=3,
            colorname=colorname,
            thickness=1
        )

    height, width = get_height_and_width(image=resized_image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 1
    lineType = 2

    if show_filename:
        bottomLeft = [5, round(0.99 * (height - 1))]
        text0 = image_pathname.replace(os.path.join(CBE_LAPTOP_DIR_GLOBAL, ''), '')
        if image_label is not None:
            text = f'{image_label}: {text0}'
        else:
            text = text0
        draw_outlined_text_on_image(
            image=resized_image,
            text=text,
            bottomLeftCornerOfText=bottomLeft,
            font=font,
            fontScale=fontScale,
            thickness=thickness,
            lineType=lineType
        )
    if extra_text is not None:
        topLeft = [5, round(0.05 * (height - 1))]
        draw_outlined_text_on_image(
            image=resized_image,
            text=extra_text,
            bottomLeftCornerOfText=topLeft,
            font=font,
            fontScale=fontScale,
            thickness=thickness,
            lineType=lineType
        )
    return resized_image


def process_vis_temperature_pair(
        vis_image_pathname,
        temperature_pickle_pathname,
        scene_datetime_object,
        nearest_vote_datetime_object,
        nearest_simplified_sensation_vote,
        nearest_sweating_vote,
        model_vis,
        model_ir,
        wait_msec,
        roi_dict,
        min_detection_confidence_percentage_vis,
        min_detection_confidence_percentage_ir,
        models_by_min_detection_confidence=None,
        use_adaptive_ir_model=False,
        draw_all_landmarks2=False,
        scale_factor_for_display=1,
        homography_method=0,
        video_writer=None,
        scene_number=None,
        lower_bound=None,
        upper_bound=None,
        save_summary=True,
        draw_temperature_reference=True,
        write_IR_image_file=True,
        show_filename=True,
        **kwargs
):
    """
    1. Render 4-panel mosaic comprising
    (A) color (VIS) image with all detected MediaPipe holistic landmarks
    (B) color (VIS) image with feature polygons and landmarks
    (C) thermal (IR) image with all detected MediaPipe holistic landmarks
    (D) thermal (IR) image with combination of transformed (VIS -> IR) polygons, IR polygons, and IR landmarks,
    plus polygon(s) locating the passive or active temperature reference(s) and summary statistics.

    2. Return summary statistics
    """

    min_detection_confidence_vis = min_detection_confidence_percentage_vis / 100.0
    min_detection_confidence_ir = min_detection_confidence_percentage_ir / 100.0
    if models_by_min_detection_confidence is not None:
        mcd_percentages = models_by_min_detection_confidence.keys()
        mcd_percentages_descending = sorted(mcd_percentages, reverse=True)
    else:
        mcds_descending = None
    if scene_number is not None:
        extra_text = f'Scene {scene_number:07}'
    else:
        extra_text = None
    vis_image0 = cv2.imread(vis_image_pathname)
    if vis_image0 is None:
        print(f'Could not load {vis_image_pathname}')
        return None
    vis_image = resize_image(image=vis_image0, scale=VIS_SCALE_GLOBAL)
    ir_image_data = \
        create_IR_image_from_temperature_pickle_file(
            pickle_pathname=temperature_pickle_pathname,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            write_IR_image_file=write_IR_image_file,
            **kwargs
        )
    if ir_image_data is None:
        print('Failed to generate IR image from ', temperature_pickle_pathname)
        return None
    ir_image_pathname, tempC_array_unbounded, ir_image_array = \
        [ir_image_data[k] for k in ['ir_image_pathname', 'tempC_array_unbounded', 'image_array']]
    ir_image0 = cv2.cvtColor(ir_image_array, cv2.COLOR_GRAY2BGR)
    ir_image = resize_image(image=ir_image0, scale=IR_SCALE_GLOBAL)
    vis_keypoints1 = get_feature_landmarks_and_polygons_from_image(image=vis_image, model=model_vis)
    vis_image1 = \
        render_features_on_image(
            image=vis_image.copy(),
            keypoints=vis_keypoints1,
            image_pathname=vis_image_pathname,
            # show_image=False,
            scale_factor_for_display=scale_factor_for_display,
            draw_face_nose_and_hand_borders=False,
            draw_face_and_hand_pose_landmarks=False,
            draw_all_landmarks=draw_all_landmarks2,
            image_label='A',
            extra_text=extra_text,
            show_filename=show_filename,
            **kwargs
        )

    vis_image2 = \
        render_features_on_image(
            image=vis_image.copy(),
            keypoints=vis_keypoints1,
            image_pathname=vis_image_pathname,
            # show_image=False,
            scale_factor_for_display=scale_factor_for_display,
            draw_face_nose_and_hand_borders=True,
            draw_face_and_hand_pose_landmarks=True,
            image_label='B',
            show_filename=show_filename,
            **kwargs
        )
    if lower_bound is None and upper_bound is None:
        extra_text = 'IR image rendering rules: none applied'
    elif upper_bound is None:
        extra_text = f'IR image rendering rules: T < {lower_bound} C set to {lower_bound} C'
    elif lower_bound is None:
        extra_text = f'IR image rendering rules: T > {upper_bound} C set to {upper_bound} C'
    else:
        extra_text = f'IR image rendering rules: T < {lower_bound} C set to {lower_bound} C, T > {upper_bound} C set to {upper_bound} C'

    if use_adaptive_ir_model and (mcd_percentages_descending is not None):
        # print(mcds_descending)
        for mcd_percentage in mcd_percentages_descending:
            mcd = mcd_percentage / 100.0
            ir_keypoints1 = get_feature_landmarks_and_polygons_from_image(image=ir_image, model=models_by_min_detection_confidence[mcd_percentage])
            pose_landmarks = ir_keypoints1['results'].pose_landmarks
            # print(mcd, pose_landmarks)
            if pose_landmarks is not None:
                break
        min_detection_confidence_ir = mcd
    else:
        ir_keypoints1 = get_feature_landmarks_and_polygons_from_image(image=ir_image, model=model_ir)
    ir_image1 = \
        render_features_on_image(
            image=ir_image.copy(),
            keypoints=ir_keypoints1,
            image_pathname=ir_image_pathname,
            # show_image=False,
            scale_factor_for_display=scale_factor_for_display,
            draw_face_nose_and_hand_borders=False,
            draw_face_and_hand_pose_landmarks=False,
            draw_all_landmarks=draw_all_landmarks2,
            image_label='C',
            extra_text=extra_text,
            show_filename=show_filename,
            **kwargs
        )
    roi_polygons = find_most_recent_roi_polygons(scene_datetime_object=scene_datetime_object, roi_dict=roi_dict)
    roi_medians = \
        compute_mean_radiometric_temperatures_of_PTR_and_ATR(
            tempC_array=tempC_array_unbounded,
            polygons=roi_polygons
        )
    # 2022-06-07 Ronnen: There may be some redundancy in the following two calls — investigate.
    reference_temperature_summary = \
        summarize_reference_temperatures(
            scene_datetime_object=scene_datetime_object,
            roi_medians=roi_medians
        )
    reference_temperature_stats = \
        get_reference_temperatures_and_offset(
            scene_datetime_object=scene_datetime_object,
            roi_medians=roi_medians
        )
    reference_temperature_offset = reference_temperature_stats['reference_temperature_offset']
    ir_keypoints1_with_transformations, homographies = \
        transform_face_nose_eyeglasses_hand_palm_polygons(
            vis_keypoints=vis_keypoints1,
            ir_keypoints=ir_keypoints1,
            homography_method=homography_method
        )
    ir_keypoints1_with_transformations_culled = \
        cull_ir_hand_and_palm_polygons(
            vis_keypoints=vis_keypoints1,
            ir_keypoints=ir_keypoints1_with_transformations
        )
    homography_summary = summarize_homographies(homographies=homographies)
    regional_temperature_stats = \
        get_regional_temperature_statistics(
            tempC_array=tempC_array_unbounded,
            ir_keypoints_dict=ir_keypoints1_with_transformations_culled,
            reference_temperature_offset=reference_temperature_offset
        )
    stat_summary = summarize_key_temperature_stats(regional_temperature_stats=regional_temperature_stats)
    visibility_summary_ir = summarize_visibilities(keypoints=ir_keypoints1_with_transformations_culled)
    visibility_stats_ir = get_visibility_stats(keypoints=ir_keypoints1_with_transformations_culled)
    min_detection_confidence_summary = f'min_detection_confidence: VIS={min_detection_confidence_vis:.2f}, IR={min_detection_confidence_ir:.2f}'
    extra_text = '\n'.join([min_detection_confidence_summary, homography_summary, stat_summary, f'IR: {visibility_summary_ir}'])
    if reference_temperature_summary is not None:
        extra_text = '\n'.join([extra_text, reference_temperature_summary])
    extra_text = '\n'.join([extra_text, f'\nsimplified sensation vote={nearest_simplified_sensation_vote}'])

    ir_image2 = \
        render_features_on_image(
            image=ir_image.copy(),
            keypoints=ir_keypoints1_with_transformations_culled,
            image_pathname=ir_image_pathname,
            # show_image=False,
            scale_factor_for_display=scale_factor_for_display,
            draw_face_nose_and_hand_borders=True,
            draw_face_and_hand_pose_landmarks=True,
            image_label='D',
            fill_polygons=False,
            extra_text=extra_text,
            show_filename=show_filename,
            **kwargs
        )
    if draw_temperature_reference:
        draw_roi_polygons_on_image(image=ir_image2, polygons=roi_polygons)
    if save_summary:
        scene_summary = \
            summarize_scene(
                scene_number=scene_number,
                scene_datetime_object=scene_datetime_object,
                nearest_simplified_sensation_vote=nearest_simplified_sensation_vote,
                nearest_sweating_vote=nearest_sweating_vote,
                nearest_vote_datetime_object=nearest_vote_datetime_object,
                homographies=homographies,
                regional_temperature_stats=regional_temperature_stats,
                reference_temperature_stats=reference_temperature_stats,
                visibility_stats_ir=visibility_stats_ir,
                min_detection_confidence_vis=min_detection_confidence_vis,
                min_detection_confidence_ir=min_detection_confidence_ir
            )
    vis_images = np.concatenate((vis_image1, vis_image2), axis=0)
    ir_images = np.concatenate((ir_image1, ir_image2), axis=0)
    mosaic = np.concatenate((vis_images, ir_images), axis=1)
    cv2.imshow('Color and thermal images with individually detected MediaPipe Holistic landmarks', mosaic)
    if video_writer is not None:
        video_writer.write(mosaic)
    key = cv2.waitKey(wait_msec)
    return key, mosaic, scene_summary


SEC_TO_MIN = 1 / 60


def process_all_trials3(
        scenes_to_process=usable_scenes_and_reference_temperatures_near_votes_global,
        roi_dict=None,
        draw_all_landmarks2=False,
        homography_method=0,
        record_video=False,
        fps=2,
        min_detection_confidence_percentage_vis=50,
        min_detection_confidence_percentage_ir=50,
        scale_factor_for_display=1,
        write_IR_image_file=True,
        static_image_mode=True,
        start_date=None,
        end_date=None,
        use_adaptive_ir_model=False,
        adaptive_percentage_step=1,
        **kwargs
):
    if roi_dict is None:
        roi_dict = load_roi_dict()
    # mcd = minimum detection confidence
    mcd_percentages = np.arange(10,51,adaptive_percentage_step)
    models_by_min_detection_confidence = \
        {mcd_percentage : create_holistic_model(min_detection_confidence=mcd_percentage / 100.0, static_image_mode=static_image_mode) for mcd_percentage in mcd_percentages}
    model_vis = models_by_min_detection_confidence[min_detection_confidence_percentage_vis]
    model_ir = models_by_min_detection_confidence[min_detection_confidence_percentage_ir]
    video_writer = None
    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    scenes_to_process_filtered = scenes_to_process.copy()
    if start_date is not None:
        scenes_to_process_filtered = scenes_to_process_filtered[scenes_to_process_filtered.index >= start_date]
    if end_date is not None:
        scenes_to_process_filtered = scenes_to_process_filtered[scenes_to_process_filtered.index <= end_date]
    if record_video:
        video_filename = f'{datetime_string}_video.mp4'
        video_pathname = os.path.join(VIDEO_DIR_GLOBAL, video_filename)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        scene_to_process = scenes_to_process_filtered.iloc[0]
        vis_image_pathname, temperature_pickle_pathname = \
            scene_to_process[['vis_image_pathname', 'temperature_pickle_pathname']]
        key, mosaic, scene_summary = \
            process_vis_temperature_pair(
                vis_image_pathname=vis_image_pathname,
                temperature_pickle_pathname=temperature_pickle_pathname,
                scene_datetime_object=None,
                scene_number=None,
                nearest_vote_datetime_object=None,
                nearest_sweating_vote=None,
                nearest_simplified_sensation_vote=None,
                model_vis=model_vis,
                model_ir=model_ir,
                models_by_min_detection_confidence=models_by_min_detection_confidence,
                use_adaptive_ir_model=use_adaptive_ir_model,
                wait_msec=1,
                roi_dict=roi_dict,
                draw_all_landmarks2=draw_all_landmarks2,
                scale_factor_for_display=scale_factor_for_display,
                draw_temperature_reference=False,
                min_detection_confidence_percentage_vis=min_detection_confidence_percentage_vis,
                min_detection_confidence_percentage_ir=min_detection_confidence_percentage_ir,
                write_IR_image_file=write_IR_image_file,
                **kwargs
            )
        height, width = get_height_and_width(image=mosaic)
        video_writer = cv2.VideoWriter(video_pathname, fourcc, fps, (width, height))
        print(f'\nRecording video to {video_pathname}')
    n = len(scenes_to_process_filtered)
    print('Scenes to process =', n)
    start_time = time.time()
    previous_time = start_time
    progress_reporting_interval = 100
    summary_filename = f'{datetime_string}_summary.csv'
    summary_pathname = os.path.join(VIDEO_DIR_GLOBAL, summary_filename)
    if os.path.exists(summary_pathname):
        os.remove(summary_pathname)
    with open(summary_pathname, mode='a') as f:
        for scene_number in range(n):
            scene_to_process = scenes_to_process_filtered.iloc[scene_number]
            vis_image_pathname, temperature_pickle_pathname, nearest_vote_datetime_object,\
            nearest_simplified_sensation_vote, nearest_sweating_vote = \
                scene_to_process[
                    ['vis_image_pathname', 'temperature_pickle_pathname', 'nearest_vote_datetime_object',
                     'nearest_simplified_sensation_vote', 'nearest_sweating_vote']
                ]
            scene_datetime_object = scene_to_process.name
            key, mosaic, scene_summary = \
                process_vis_temperature_pair(
                    vis_image_pathname=vis_image_pathname,
                    temperature_pickle_pathname=temperature_pickle_pathname,
                    scene_datetime_object=scene_datetime_object,
                    nearest_vote_datetime_object=nearest_vote_datetime_object,
                    nearest_simplified_sensation_vote=nearest_simplified_sensation_vote,
                    nearest_sweating_vote=nearest_sweating_vote,
                    model_vis=model_vis,
                    model_ir=model_ir,
                    roi_dict=roi_dict,
                    wait_msec=1,
                    draw_all_landmarks2=draw_all_landmarks2,
                    scale_factor_for_display=scale_factor_for_display,
                    homography_method=homography_method,
                    # show_polygon=True,
                    show_filename=True,
                    video_writer=video_writer,
                    scene_number=scene_number,
                    write_IR_image_file=write_IR_image_file,
                    min_detection_confidence_percentage_vis=min_detection_confidence_percentage_vis,
                    min_detection_confidence_percentage_ir=min_detection_confidence_percentage_ir,
                    models_by_min_detection_confidence=models_by_min_detection_confidence,
                    use_adaptive_ir_model=use_adaptive_ir_model,
                    **kwargs)
            df_scene_summary = pd.DataFrame([scene_summary]).set_index('scene_datetime_object')
            header = f.tell() == 0 # Include header only when writing first line to CSV summary file
            df_scene_summary.to_csv(f, header=header, na_rep='#N/A', line_terminator='\n')
            if (scene_number > 0) and (scene_number % progress_reporting_interval == 0):
                current_time = time.time()
                recent_elapsed_time = current_time - previous_time
                recent_time_per_scene = recent_elapsed_time / progress_reporting_interval
                scenes_remaining = (n - 1) - scene_number
                estimated_time_remaining = scenes_remaining * recent_time_per_scene
                print(f'\nScenes processed = {scene_number+1}; scenes remaining = {scenes_remaining} (out of {n})')
                print(
                    f'Estimated time remaining: {estimated_time_remaining * SEC_TO_MIN:.1f} min (recent time per scene = {recent_time_per_scene:.2f} s)')
                previous_time = current_time

            if key > 0:
                char = chr(key)
                if char in ['p', SPACE_GLOBAL]:
                    cv2.waitKey(0)
                elif char in ['q', ESC_GLOBAL]:
                    break
    if record_video and video_writer is not None:
        video_writer.release()
        print(f'\nRecorded video to {video_pathname}')
    elapsed_time = time.time() - start_time
    time_per_scene = elapsed_time / (scene_number+1)

    print(f'\nScenes processed = {scene_number+1}\nElapsed time = {elapsed_time * SEC_TO_MIN:.1f} min\nAverage time per scene = {time_per_scene:.2f} s')
    cv2.destroyAllWindows()
    print('Wrote ', summary_pathname)
    return model_vis, model_ir


my_results_global = None


