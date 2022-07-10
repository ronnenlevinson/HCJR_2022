"""

"""

# Import packages


from load_past_trial_data import usable_scenes_and_reference_temperatures_near_votes_global
from scene_analysis import process_all_trials3


# if False:
#     mark_temperature_references()


if True:
    model_vis, model_ir = \
        my_df_scene_summary = process_all_trials3(
            scenes_to_process=usable_scenes_and_reference_temperatures_near_votes_global,
            # start_date='2022-06-01',
            # end_date='2022-06-15 16:00:00',
            use_adaptive_ir_model=True,
            # verbose=True,
            draw_all_landmarks2=True,
            scale_factor_for_display=0.9,
            record_video=True,
            homography_method=0,  # 0, cv2.RANSAC, cv2.LMEDS, or cv2.RHO
            # regenerate_ir_images=True,
            fps=5,
            lower_bound=None,  # SKIN_TEMPERATURE_LOWER_BOUND_C,
            upper_bound=None, # SKIN_TEMPERATURE_UPPER_BOUND_C,
            # show_extraction=False,
            min_detection_confidence_percentage_vis=50,
            min_detection_confidence_percentage_ir=10,
            adaptive_percentage_step=5,
            write_IR_image_file=True,
            static_image_mode=True
    )
    print(model_vis)

#
# def fred(x):
#     return x+1

if False:
    correct_datetimes_formats_in_filenames_and_foldernames(CBE_LAPTOP_DIR_GLOBAL, deep=True, implement=True, start_date=pd.Timestamp('2022-01-01'))

print('done')