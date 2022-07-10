import time

import numpy as np


def inclusive_range(start, stop, step=1):
    """
    Return range (numpy array) from start to stop by step, including stop value.
    """
    return np.arange(start, stop + step, step)

def report_elapsed_time(start_time, digits=1):
    """
    Print time elapsed (sec) since start_time.
    """
    elapsed_time = time.time() - start_time
    print(f'Elapsed time = {elapsed_time:.{digits}f} sec')






