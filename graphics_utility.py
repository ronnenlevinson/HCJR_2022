import cv2
import matplotlib
import numpy as np
import pandas as pd

from config import IR_SCALE_GLOBAL


def get_height_and_width(image):
    """
    Return height and width of image as tuple.
    Works for both grayscale images (shape = height, width) and color images (shape = height, width, byte depth).
    """
    height, width = image.shape[0:2]
    return height, width


def close_polygon(points):
    """
    If the last point (vertex) of a polygon is not the same as its first point, append the first point
    to the array of vertices.
    """
    points_array = np.array(points)
    # Can't close a polygon if it has less than 3 vertices.
    if len(points_array) > 2:
        first_point = points_array[0]
        last_point = points_array[-1]
        if not np.array_equal(last_point, first_point):
            points_array = np.append(points_array, [first_point], axis=0)
    return points_array


def resize_image(image, scale=1):
    """
    Resize image geometry by multiplicative factor 'scale'.
    """
    height, width = get_height_and_width(image=image)
    new_height = int(height * scale)
    new_width = int(width * scale)
    image_resized = cv2.resize(image, (new_width, new_height))
    return image_resized


def convert_normalized_coordinates_to_pixel_coordinates(image, normalized_points):
    """
    Convert array of float normalized coordinates [x:0-1, y:0-1]
    to array of integer pixel coordinates [x:0-(pixel width-1), y:0-(pixel height-1)].
    """
    height, width = get_height_and_width(image=image)
    pixel_points = np.array([(round(x * (width - 1)), round(y * (height - 1))) for x, y in normalized_points], np.int32)
    return pixel_points


def convert_pixel_coordinates_to_normalized_coordinates(image, pixel_points):
    """
    Convert array of integer pixel coordinates [x:0-(pixel width-1), y:0-(pixel height-1)]
    to array of float normalized coordinates [x:0-1, y:0-1].
    """
    height, width = get_height_and_width(image=image)
    normalized_points = np.array([(x / width, y / height) for x, y in pixel_points], float)
    return normalized_points


def colorname_to_bgr(name):
    """
    Return blue, green, red color values corresponding to a color name.
    """
    r, g, b = [round(255 * x) for x in matplotlib.colors.to_rgb(name)]
    # For silly historical reasons, OpenCV uses blue-green-red (BGR) rather than red-green-blue (RGB) color coordinates.
    # <https://learnopencv.com/why-does-opencv-use-bgr-color-format/>
    return b, g, r


def draw_outlined_text_on_image(
        image,
        text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        thickness,
        lineType,
        line_spacing_pixels=25
):
    """
    Draw outlined text (white over black) in OpenCV.
    """
    if text is None:
        return None
    text_list = text.split('\n')
    n_lines = len(text_list)
    # Handle multiline text by applying this function to each text line,
    # moving each line down by a given number of pixels.
    if n_lines > 1:
        bottomLeftCornerOfText_adj = bottomLeftCornerOfText
        for text_line in text_list:
            draw_outlined_text_on_image(
                image=image,
                text=text_line,
                bottomLeftCornerOfText=bottomLeftCornerOfText_adj,
                font=font,
                fontScale=fontScale,
                thickness=thickness,
                lineType=lineType
            )
            bottomLeftCornerOfText_adj[1] += line_spacing_pixels
    else:
        white = colorname_to_bgr('white')
        black = colorname_to_bgr('black')
        # Note that cv2 routines tend to expect points as a tuple rather than a list or an array.
        # Draw thick black text
        cv2.putText(
            image,
            text,
            tuple(bottomLeftCornerOfText),
            font,
            fontScale,
            black,
            2 * thickness,
            lineType
        )
        # Draw thin white text over thick black text to create outlined text
        cv2.putText(
            image,
            text,
            tuple(bottomLeftCornerOfText),
            font,
            fontScale,
            white,
            thickness,
            lineType
        )


def draw_polygon_on_image(
        image,
        polygon=None,  # float normalized coordinates
        pixel_polygon=None,  # integer pixel coordinates
        colorname='green',
        thickness=1,
        markerSize=3,
        markerType=cv2.MARKER_DIAMOND,
        show_vertices=False
):
    """
    Draw polygon on image.
    """
    # Landmark vertices and thus polygons generated with MediaPipe use normalized coordinates while OpenCV image drawing
    # routines such as cv2.polylines() and cv2.marker() expect integer pixel coordinates. We use the term 'polygon'
    # to refer to an array of float normalized coordinates and the term 'pixel polygon' to refer to an array of
    # integer pixel coordinates.
    if polygon is None and pixel_polygon is None:
        return None
    elif polygon is not None:
        points = convert_normalized_coordinates_to_pixel_coordinates(image=image, normalized_points=polygon)
    elif pixel_polygon is not None:
        points = pixel_polygon
    points_reshaped = points.reshape((-1, 1, 2))
    isClosed = True
    color = colorname_to_bgr(colorname)
    image = cv2.polylines(image, [points_reshaped], isClosed, color, thickness)
    if show_vertices:
        for point in points:
            # OpenCV drawing routines tend to expect individual points as a tuple.
            cv2.drawMarker(image, tuple(point), color, markerType=markerType, markerSize=markerSize)
    # return image


def draw_circles_on_image(
        image,
        centers,  # float normalized coordinates
        radius,  # integer pixels
        colorname,
        thickness
):
    """
    Draw circles on image.
    """
    if centers is not None:
        color = colorname_to_bgr(colorname)
        points = convert_normalized_coordinates_to_pixel_coordinates(image=image, normalized_points=centers)
        for point in points:
            # OpenCV drawing routines tend to expect individual points as a tuple.
            cv2.circle(image, tuple(point), radius, color, thickness)
    # return image

# The following mask functions are to be used to calculate statistics on regions of the thermal image
# that map to corresponding region in the color image.

def create_mask_inside_polygon(
        image,
        polygon  # float normalized coordinates
):
    """
    Return raster mask of grayscale image that is True inside the convex polygon bounded by the supplied vertices
    and False elsewhere. The vertices should be a list or numpy array of float normalized (x,y) coordinates.
    """
    height, width = get_height_and_width(image=image)
    arr = np.zeros((height, width))
    closed_polygon = close_polygon(points=polygon)  # Might not be needed — check later

    # integer pixel coordinates
    pixel_polygon = \
        convert_normalized_coordinates_to_pixel_coordinates(
            image=image,
            normalized_points=closed_polygon
        )
    # If the polygon is not convex, we could use fillPoly() instead, but the latter is reputed to be much slower.
    # The only polygon that might be concave is that for the eyeglasses region.
    cv2.fillConvexPoly(arr, pixel_polygon, 1)
    mask = pd.DataFrame(arr == 1)
    return mask


def draw_filled_polygon_on_image(
        image,
        polygon,
        colorname=None,
        verbose=False
):
    """
    Draw filled polygon on image.
    """
    if colorname is None:
        print('Must specify value or colorname')
        return None
    value = colorname_to_bgr(colorname)
    if polygon is not None:
        mask = create_mask_inside_polygon(image=image, polygon=polygon)
        image[mask] = value
        if verbose:
            cv2.imshow('Image with filled polygon', image)
            cv2.waitKey(1000)
    # return image


def extract_raster_values_inside_polygon(
        raster,
        polygon,
        extraction_colorname='red',
        show_extraction=False,
        visualization_rescale=IR_SCALE_GLOBAL
):
    """
    Extract raster values inside polygon to a 1-D numpy array.
    """
    if polygon is None:
        return None
    mask = create_mask_inside_polygon(image=raster, polygon=polygon)
    result = raster[mask]
    if show_extraction:
        resized_raster = resize_image(image=raster, scale=visualization_rescale)
        grayscale_raster = rescale_raster_values(raster=resized_raster, new_min=0, new_max=255, new_type=np.uint8)
        color_raster = cv2.cvtColor(grayscale_raster, cv2.COLOR_GRAY2RGB)
        draw_filled_polygon_on_image(
            image=color_raster,
            polygon=polygon,
            colorname=extraction_colorname
        )
        cv2.imshow('Temperature pixel extraction', color_raster)
        cv2.waitKey(200)
    return result


def extract_raster_array_values_inside_multiple_polygons(
        raster,
        polygon_dict  # float normalized coordinates
):
    """
    Extract raster values inside polygons in a dictionary, yielding a dictionary of 1-D numpy arrays.
    """
    result = \
        {k: extract_raster_values_inside_polygon(raster=raster, polygon=v) \
         for k, v in polygon_dict.items()
         }
    return result


def is_polygon_A_inside_polygon_B(
        polygon_A,  # float normalized coordinates
        polygon_B,  # float normalized coordinates
        image
):
    """
    Determine whether every vertex of polygon A lies inside polygon B.
    """
    if len(polygon_B) < 3:
        return False
    else:
        pixel_polygon_A = convert_normalized_coordinates_to_pixel_coordinates(image=image, normalized_points=polygon_A)
        pixel_polygon_B = convert_normalized_coordinates_to_pixel_coordinates(image=image, normalized_points=polygon_B)
        # cv2.pointPolygonTest() seems to expect integer-coordinate contours but float values for the point
        are_all_points_inside = \
            all(
                [cv2.pointPolygonTest(pixel_polygon_B, tuple(point.astype(float)), False) == 1 \
                 for point in pixel_polygon_A]
            )
        return are_all_points_inside


def extract_raster_values_between_two_nested_polygons(
        raster,
        inner_polygon,
        outer_polygon
):
    """
    Extract to a 1-D numpy array all raster values contained between two nested polygons.
    """
    if inner_polygon is None or outer_polygon is None:
        return None
    inner_polygon_is_inside_outer_polygon = \
        is_polygon_A_inside_polygon_B(
            polygon_A=inner_polygon,
            polygon_B=outer_polygon,
            image=raster
        )
    if not inner_polygon_is_inside_outer_polygon:
        print('Inner polygon is not inside outer polygon')
        return None
    inner_mask = create_mask_inside_polygon(image=raster, polygon=inner_polygon)
    outer_mask = create_mask_inside_polygon(image=raster, polygon=outer_polygon)
    # Raster value type must be float to replace values with np.nan
    raster1 = np.array(raster, dtype=float)  # was raster.copy()
    # Set values inside inner polygon to NaN
    raster1[inner_mask] = np.nan
    # Get 1-D numpy array of all values within outer polygon
    raw_values = raster1[outer_mask]
    # Remove all NaN values
    values = raw_values[~np.isnan(raw_values)]
    return values


def rescale_raster_values(raster, new_min, new_max, new_type=None):
    """
    Linearly rescale raster values mapping current min to new_min and current max to new_max,
    converting raster to new_type if the latter is specified.
    """
    raster_min = raster.min()
    raster_max = raster.max()
    new_raster = new_min + (raster - raster_min) * (new_max - new_min) / (raster_max - raster_min)
    if new_type is not None:
        new_raster = new_raster.astype(new_type)
    return new_raster
