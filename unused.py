import glob
import os


def image_pathnames(image_dir, spectrum='VIS'):
    """
    Return paths to all VIS or IR image files with a directory.
    2022-06-05 Ronnen: We no longer read IR images from file since they are always
    regenerated from temperature rasters.
    """
    if spectrum == 'VIS':
        pattern = 'vis_*'
    elif spectrum == 'IR':
        pattern = 'ir_*'
    else:
        print('Unknown spectrum: ', spectrum)
        return None
    names = glob.glob(os.path.join(image_dir, pattern))
    names.sort()
    return names


def get_temperature_pickle_pathnames(pickle_dir):
    """
    Return paths to all temperature raster pickle files with a directory.
    MIGHT NOT BE USED
    """
    pattern = 'temp_info_*.pickle'
    names = glob.glob(os.path.join(pickle_dir, pattern))
    names.sort()
    return names
