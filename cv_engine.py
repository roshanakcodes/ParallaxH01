import cv2
import numpy as np
import requests

CAMERA_URL = "http://172.16.44.76:8080/shot.jpg"

def capture_image(camera_url=CAMERA_URL):
    """
    Capture one image from the friend's IP camera.

    Returns:
        BGR OpenCV image
        None if capture fails
    """

    try:
        response = requests.get(
            camera_url,
            timeout=10
        )

        response.raise_for_status()

        image_array = np.frombuffer(
            response.content,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception as e:

        print(
            f"Camera capture error: {e}"
        )

        return None


# ============================================================
# CONTOUR NORMALIZATION
# ============================================================

def normalize_contour(contour):
    """
    Convert contour into the OpenCV format:

        N x 1 x 2

    using int32.
    """

    if contour is None:
        return None

    contour = np.asarray(
        contour,
        dtype=np.int32
    )

    if contour.size == 0:
        return None

    if contour.ndim == 2:

        if contour.shape[1] != 2:
            return None

        contour = contour.reshape(
            (-1, 1, 2)
        )

    elif contour.ndim == 3:

        if (
            contour.shape[1] != 1
            or contour.shape[2] != 2
        ):
            return None

    else:
        return None

    if len(contour) < 2:
        return None

    return np.ascontiguousarray(
        contour,
        dtype=np.int32
    )


# ============================================================
# GRID DETECTION
# ============================================================

def detect_grid_color(image):
    """
    Detect the green grid lines.

    White = grid
    Black = everything else
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower_green = np.array(
        [30, 25, 25],
        dtype=np.uint8
    )

    upper_green = np.array(
        [100, 255, 230],
        dtype=np.uint8
    )

    grid_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    return grid_mask


# ============================================================
# MISSING GRID DETECTION
# ============================================================

def find_missing_grid_region(image):
    """
    The pill covers the grid.

    Therefore, the area occupied by the pill has
    significantly lower grid-line density.
    """

    grid_mask = detect_grid_color(
        image
    )

    kernel_size = 51

    kernel = np.ones(
        (
            kernel_size,
            kernel_size
        ),
        dtype=np.float32
    )

    grid_float = (
        grid_mask.astype(
            np.float32
        ) / 255.0
    )

    density = cv2.filter2D(
        grid_float,
        -1,
        kernel / (
            kernel_size * kernel_size
        )
    )

    object_mask = (
        density < 0.025
    ).astype(
        np.uint8
    ) * 255

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    object_mask = cv2.morphologyEx(
        object_mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    object_mask = cv2.morphologyEx(
        object_mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    return (
        object_mask,
        grid_mask
    )


# ============================================================
# ORANGE PILL DETECTION
# ============================================================

def detect_orange_object(image):
    """
    Detect the orange pill.

    This is used as strong colour evidence.

    Grid interruption is still available as a fallback.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower = np.array(
        [3, 70, 70],
        dtype=np.uint8
    )

    upper = np.array(
        [35, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


# ============================================================
# CONTOUR VALIDATION
# ============================================================

def contour_is_valid(
    contour,
    image_shape
):
    """
    Reject obviously invalid pill candidates.
    """

    contour = normalize_contour(
        contour
    )

    if contour is None:
        return False

    area = cv2.contourArea(
        contour
    )

    if area < 250:
        return False

    image_h, image_w = (
        image_shape[:2]
    )

    image_area = (
        image_h * image_w
    )

    if area > image_area * 0.05:
        return False

    x, y, w, h = cv2.boundingRect(
        contour
    )

    if w > image_w * 0.30:
        return False

    if h > image_h * 0.30:
        return False

    if w < 20 or h < 10:
        return False

    rect = cv2.minAreaRect(
        contour
    )

    rw, rh = rect[1]

    if rw <= 0 or rh <= 0:
        return False

    aspect = (
        max(rw, rh) /
        min(rw, rh)
    )

    if aspect < 1.35:
        return False

    if aspect > 4.5:
        return False

    return True


# ============================================================
# ORANGE SCORE
# ============================================================

def contour_colour_score(
    contour,
    hsv
):
    """
    Calculate how strongly a contour
    contains orange pixels.
    """

    contour = normalize_contour(
        contour
    )

    if contour is None:
        return 0.0

    candidate_mask = np.zeros(
        hsv.shape[:2],
        dtype=np.uint8
    )

    cv2.drawContours(
        candidate_mask,
        [contour],
        -1,
        255,
        -1
    )

    pixels = (
        candidate_mask > 0
    )

    if not np.any(pixels):
        return 0.0

    hue = hsv[:, :, 0][pixels]
    saturation = hsv[:, :, 1][pixels]
    value = hsv[:, :, 2][pixels]

    orange = (
        (hue >= 3) &
        (hue <= 35) &
        (saturation > 60) &
        (value > 70)
    )

    return (
        float(np.count_nonzero(orange))
        /
        float(len(hue))
    )


# ============================================================
# FIND PILL REGION
# ============================================================

def find_pill_region(
    object_mask,
    image
):
    """
    Find the pill.

    Priority:

    1. Orange colour detection
    2. Missing-grid detection
    3. Combined fallback
    """

    if image is None:
        return None

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    image_area = (
        image.shape[0] *
        image.shape[1]
    )

    # --------------------------------------------------------
    # 1. Orange mask
    # --------------------------------------------------------

    orange_mask = detect_orange_object(
        image
    )

    contours, _ = cv2.findContours(
        orange_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        contour = normalize_contour(
            contour
        )

        if not contour_is_valid(
            contour,
            image.shape
        ):
            continue

        area = cv2.contourArea(
            contour
        )

        rect = cv2.minAreaRect(
            contour
        )

        rw, rh = rect[1]

        length = max(
            rw,
            rh
        )

        width = min(
            rw,
            rh
        )

        if width <= 0:
            continue

        aspect = (
            length / width
        )

        rect_area = (
            length * width
        )

        fill_ratio = (
            area /
            max(rect_area, 1.0)
        )

        if fill_ratio < 0.45:
            continue

        orange_ratio = (
            contour_colour_score(
                contour,
                hsv
            )
        )

        score = 0.0

        # Strong orange evidence
        score += (
            orange_ratio * 1000
        )

        # Prefer normal pill aspect ratios
        if 1.5 <= aspect <= 3.5:
            score += 50

        elif 1.35 <= aspect <= 4.5:
            score += 15

        # Prefer filled objects
        score += (
            min(
                fill_ratio,
                1.0
            ) * 80
        )

        # Moderate area
        score += min(
            area / 1000.0,
            40
        )

        candidates.append(
            (
                score,
                contour
            )
        )

    if candidates:

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return candidates[0][1]

    # --------------------------------------------------------
    # 2. Missing-grid fallback
    # --------------------------------------------------------

    if object_mask is not None:

        contours, _ = cv2.findContours(
            object_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        fallback = []

        for contour in contours:

            contour = normalize_contour(
                contour
            )

            if not contour_is_valid(
                contour,
                image.shape
            ):
                continue

            area = cv2.contourArea(
                contour
            )

            fallback.append(
                (
                    area,
                    contour
                )
            )

        if fallback:

            fallback.sort(
                key=lambda item: item[0],
                reverse=True
            )

            return fallback[0][1]

    # --------------------------------------------------------
    # 3. Combined fallback
    # --------------------------------------------------------

    if object_mask is not None:

        combined = cv2.bitwise_or(
            object_mask,
            orange_mask
        )

        contours, _ = cv2.findContours(
            combined,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        fallback = []

        for contour in contours:

            contour = normalize_contour(
                contour
            )

            if not contour_is_valid(
                contour,
                image.shape
            ):
                continue

            area = cv2.contourArea(
                contour
            )

            fallback.append(
                (
                    area,
                    contour
                )
            )

        if fallback:

            fallback.sort(
                key=lambda item: item[0],
                reverse=True
            )

            return fallback[0][1]

    return None


# ============================================================
# COLOUR DETECTION
# ============================================================

HUE_RANGES = {

    "RED": [
        (0, 10),
        (170, 179)
    ],

    "ORANGE": [
        (11, 20)
    ],

    "YELLOW": [
        (21, 33)
    ],

    "GREEN": [
        (34, 85)
    ],

    "BLUE": [
        (86, 125)
    ],

    "PURPLE": [
        (126, 145)
    ],

    "PINK": [
        (146, 169)
    ]
}


def detect_colour(
    image,
    contour
):
    """
    Determine pill colour from pixels inside contour.
    """

    contour = normalize_contour(
        contour
    )

    if contour is None:
        return "UNKNOWN"

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )

    cv2.drawContours(
        mask,
        [contour],
        -1,
        255,
        -1
    )

    pixels = hsv[
        mask == 255
    ]

    if pixels.size == 0:
        return "UNKNOWN"

    mean_h, mean_s, mean_v = (
        pixels.mean(axis=0)
    )

    # Achromatic colours
    if mean_s < 40:

        if mean_v > 180:
            return "WHITE"

        if mean_v < 60:
            return "BLACK"

        return "GREY"

    for name, ranges in HUE_RANGES.items():

        for lo, hi in ranges:

            if lo <= mean_h <= hi:
                return name

    return "UNKNOWN"


# ============================================================
# SHAPE FEATURES
# ============================================================

def shape_features(
    contour
):
    """
    Calculate geometric features.
    """

    contour = normalize_contour(
        contour
    )

    if contour is None:
        return None

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    x, y, w, h = cv2.boundingRect(
        contour
    )

    if min(w, h) > 0:

        aspect_ratio = (
            max(w, h) /
            min(w, h)
        )

    else:

        aspect_ratio = 0

    if perimeter > 0:

        circularity = (
            4 *
            np.pi *
            area /
            (perimeter ** 2)
        )

    else:

        circularity = 0

    if w * h > 0:

        extent = (
            area /
            (w * h)
        )

    else:

        extent = 0

    return {

        "area": float(area),

        "perimeter": float(
            perimeter
        ),

        "width": int(w),

        "height": int(h),

        "aspect_ratio": float(
            aspect_ratio
        ),

        "circularity": float(
            circularity
        ),

        "extent": float(
            extent
        )
    }


# ============================================================
# SHAPE CLASSIFICATION
# ============================================================

def classify_shape(
    contour
):
    """
    Classify pill shape using:

    - extent
    - aspect ratio
    - circularity
    """

    features = shape_features(
        contour
    )

    if features is None:

        return (
            "IRREGULAR",
            {}
        )

    aspect = features[
        "aspect_ratio"
    ]

    circularity = features[
        "circularity"
    ]

    extent = features[
        "extent"
    ]

    # Rectangular
    if (
        extent > 0.85
        and aspect < 1.4
    ):

        return (
            "RECTANGULAR",
            features
        )

    # Round
    if (
        circularity > 0.85
        and aspect < 1.15
    ):

        return (
            "ROUND",
            features
        )

    # Capsule
    if (
        aspect >= 1.6
        and extent > 0.75
    ):

        return (
            "CAPSULE",
            features
        )

    # Oval
    if aspect >= 1.15:

        return (
            "OVAL",
            features
        )

    return (
        "IRREGULAR",
        features
    )


# ============================================================
# PIXEL MEASUREMENT
# ============================================================

def measure_pill_pixels(
    contour
):
    """
    Measure pill dimensions in pixels
    using a minimum-area rectangle.
    """

    contour = normalize_contour(
        contour
    )

    if contour is None:
        return None

    rect = cv2.minAreaRect(
        contour
    )

    rw, rh = rect[1]

    if rw <= 0 or rh <= 0:
        return None

    length = max(
        rw,
        rh
    )

    width = min(
        rw,
        rh
    )

    return {

        "length_px": float(
            length
        ),

        "width_px": float(
            width
        ),

        "angle": float(
            rect[2]
        ),

        "rect": rect
    }


# ============================================================
# GRID PEAK DETECTION
# ============================================================

def smooth_1d(
    signal,
    window=3
):

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    if len(signal) < window:
        return signal

    kernel = (
        np.ones(window)
        /
        float(window)
    )

    return np.convolve(
        signal,
        kernel,
        mode="same"
    )


def find_grid_peaks(
    projection
):
    """
    Find repeated grid line positions.
    """

    projection = smooth_1d(
        projection,
        3
    )

    if len(projection) < 20:
        return []

    maximum = float(
        np.max(projection)
    )

    if maximum <= 0:
        return []

    threshold = max(
        5.0,
        maximum * 0.12
    )

    raw = []

    for i in range(
        1,
        len(projection) - 1
    ):

        if projection[i] < threshold:
            continue

        if (
            projection[i]
            >= projection[i - 1]
            and
            projection[i]
            >= projection[i + 1]
        ):

            raw.append(i)

    if not raw:
        return []

    merged = []

    for p in raw:

        if not merged:

            merged.append(p)
            continue

        if (
            p - merged[-1]
            <= 3
        ):

            merged[-1] = int(
                round(
                    (
                        merged[-1]
                        + p
                    ) / 2.0
                )
            )

        else:

            merged.append(p)

    return merged


# ============================================================
# GRID SPACING
# ============================================================

def estimate_spacing_from_lines(
    lines
):
    """
    Estimate local grid spacing.

    The small repeating grid spacing is
    treated as approximately 1 mm.
    """

    if len(lines) < 4:
        return None

    differences = np.diff(
        np.asarray(
            lines,
            dtype=np.float32
        )
    )

    differences = differences[
        (differences >= 3.0)
        &
        (differences <= 20.0)
    ]

    if len(differences) < 2:
        return None

    rounded = np.round(
        differences
    ).astype(np.int32)

    values, counts = np.unique(
        rounded,
        return_counts=True
    )

    if len(values) == 0:
        return None

    best_value = values[
        np.argmax(counts)
    ]

    selected = differences[
        np.abs(
            differences -
            best_value
        ) <= 1.5
    ]

    if len(selected) == 0:
        return None

    return float(
        np.median(selected)
    )


# ============================================================
# LOCAL GRID SCALE
# ============================================================

def estimate_local_grid_scales(
    image,
    contour
):
    """
    Estimate X and Y pixels/mm near the pill.

    IMPORTANT:
    There is NO global fixed scale here.

    The scale is calculated from the current image
    around the current pill.
    """

    contour = normalize_contour(
        contour
    )

    if contour is None:
        return None

    grid_mask = detect_grid_color(
        image
    )

    x, y, bw, bh = cv2.boundingRect(
        contour
    )

    h, w = image.shape[:2]

    # Local area around pill
    margin_x = max(
        120,
        int(bw * 4)
    )

    margin_y = max(
        120,
        int(bh * 4)
    )

    x1 = max(
        0,
        x - margin_x
    )

    y1 = max(
        0,
        y - margin_y
    )

    x2 = min(
        w,
        x + bw + margin_x
    )

    y2 = min(
        h,
        y + bh + margin_y
    )

    local = grid_mask[
        y1:y2,
        x1:x2
    ]

    if local.size == 0:
        return None

    # --------------------------------------------------------
    # Vertical grid lines
    # --------------------------------------------------------

    vertical_projection = np.sum(
        local > 0,
        axis=0
    ).astype(
        np.float32
    )

    vertical_lines = find_grid_peaks(
        vertical_projection
    )

    # --------------------------------------------------------
    # Horizontal grid lines
    # --------------------------------------------------------

    horizontal_projection = np.sum(
        local > 0,
        axis=1
    ).astype(
        np.float32
    )

    horizontal_lines = find_grid_peaks(
        horizontal_projection
    )

    x_spacing = (
        estimate_spacing_from_lines(
            vertical_lines
        )
    )

    y_spacing = (
        estimate_spacing_from_lines(
            horizontal_lines
        )
    )

    return (
        x_spacing,
        y_spacing
    )


# ============================================================
# MILLIMETER MEASUREMENT
# ============================================================

def measure_pill_mm(
    image,
    contour
):
    """
    Measure pill dimensions in millimetres
    using LOCAL grid scale.
    """

    contour = normalize_contour(
        contour
    )

    if contour is None:
        return None

    pixel_measurement = (
        measure_pill_pixels(
            contour
        )
    )

    if pixel_measurement is None:
        return None

    length_px = pixel_measurement[
        "length_px"
    ]

    width_px = pixel_measurement[
        "width_px"
    ]

    angle = pixel_measurement[
        "angle"
    ]

    scales = (
        estimate_local_grid_scales(
            image,
            contour
        )
    )

    if scales is None:
        return None

    x_scale, y_scale = scales

    if (
        x_scale is None
        and
        y_scale is None
    ):
        return None

    # If one direction failed,
    # use the other direction.
    if x_scale is None:
        x_scale = y_scale

    if y_scale is None:
        y_scale = x_scale

    if (
        x_scale <= 0
        or
        y_scale <= 0
    ):
        return None

    # --------------------------------------------------------
    # Convert dimensions
    # --------------------------------------------------------

    angle_abs = abs(
        angle
    )

    if angle_abs > 45:

        angle_abs = (
            90 -
            angle_abs
        )

    if angle_abs < 15:

        length_mm = (
            length_px /
            x_scale
        )

        width_mm = (
            width_px /
            y_scale
        )

    else:

        scale = (
            float(x_scale)
            +
            float(y_scale)
        ) / 2.0

        length_mm = (
            length_px /
            scale
        )

        width_mm = (
            width_px /
            scale
        )

    local_scale = (
        float(x_scale)
        +
        float(y_scale)
    ) / 2.0

    return {

        "length_px": float(
            length_px
        ),

        "width_px": float(
            width_px
        ),

        "angle": float(
            angle
        ),

        "length_mm": float(
            length_mm
        ),

        "width_mm": float(
            width_mm
        ),

        "pixels_per_mm": float(
            local_scale
        ),

        "x_pixels_per_mm": float(
            x_scale
        ),

        "y_pixels_per_mm": float(
            y_scale
        )
    }


# ============================================================
# DRAW RESULT
# ============================================================

def draw_result(
    image,
    contour
):
    """
    Draw detected contour and
    minimum-area rectangle.
    """

    output = image.copy()

    contour = normalize_contour(
        contour
    )

    if contour is None:
        return output

    rect = cv2.minAreaRect(
        contour
    )

    box = cv2.boxPoints(
        rect
    )

    box = np.int32(
        np.round(box)
    )

    # Red = measurement rectangle
    cv2.drawContours(
        output,
        [box],
        0,
        (0, 0, 255),
        3
    )

    # Green = actual contour
    cv2.drawContours(
        output,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    # Blue = centre
    cx, cy = rect[0]

    cv2.circle(
        output,
        (
            int(round(cx)),
            int(round(cy))
        ),
        4,
        (255, 0, 0),
        -1
    )

    return output


# ============================================================
# COMPLETE CV ANALYSIS
# ============================================================

def analyze_image(image):
    """
    Run the complete CV pipeline on an image.

    Returns a JSON-friendly dictionary.
    """

    if image is None:

        return {
            "status": "ERROR",
            "message": "Image is None"
        }

    # --------------------------------------------------------
    # 1. Detect missing grid
    # --------------------------------------------------------

    object_mask, grid_mask = (
        find_missing_grid_region(
            image
        )
    )

    # --------------------------------------------------------
    # 2. Detect pill
    # --------------------------------------------------------

    contour = find_pill_region(
        object_mask,
        image
    )

    if contour is None:

        return {
            "status": "FAIL",
            "message": "Pill could not be detected"
        }

    contour = normalize_contour(
        contour
    )

    if contour is None:

        return {
            "status": "FAIL",
            "message": "Invalid pill contour"
        }

    # --------------------------------------------------------
    # 3. Colour
    # --------------------------------------------------------

    colour = detect_colour(
        image,
        contour
    )

    # --------------------------------------------------------
    # 4. Shape
    # --------------------------------------------------------

    shape, features = (
        classify_shape(
            contour
        )
    )

    # --------------------------------------------------------
    # 5. Pixel dimensions
    # --------------------------------------------------------

    pixel_measurement = (
        measure_pill_pixels(
            contour
        )
    )

    if pixel_measurement is None:

        return {
            "status": "FAIL",
            "message": "Could not measure pill pixels"
        }

    # --------------------------------------------------------
    # 6. Millimetre dimensions
    # --------------------------------------------------------

    mm_measurement = (
        measure_pill_mm(
            image,
            contour
        )
    )

    if mm_measurement is None:

        return {
            "status": "FAIL",
            "message": "Could not determine local grid scale"
        }

    # --------------------------------------------------------
    # 7. Final result
    # --------------------------------------------------------

    result = {

        "status": "SUCCESS",

        "color": colour,

        "shape": shape,

        # Keep this field because your
        # Flask/backend structure currently
        # expects a single size_mm value.
        #
        # We use the major dimension.
        "size_mm": round(
            mm_measurement[
                "length_mm"
            ],
            2
        ),

        # More complete dimensional data
        "length_mm": round(
            mm_measurement[
                "length_mm"
            ],
            2
        ),

        "width_mm": round(
            mm_measurement[
                "width_mm"
            ],
            2
        ),

        "length_px": round(
            mm_measurement[
                "length_px"
            ],
            2
        ),

        "width_px": round(
            mm_measurement[
                "width_px"
            ],
            2
        ),

        "angle": round(
            mm_measurement[
                "angle"
            ],
            2
        ),

        # LOCAL scale
        "pixels_per_mm": round(
            mm_measurement[
                "pixels_per_mm"
            ],
            3
        ),

        "x_pixels_per_mm": round(
            mm_measurement[
                "x_pixels_per_mm"
            ],
            3
        ),

        "y_pixels_per_mm": round(
            mm_measurement[
                "y_pixels_per_mm"
            ],
            3
        ),

        # Shape features
        "circularity": round(
            features.get(
                "circularity",
                0
            ),
            4
        ),

        "aspect_ratio": round(
            features.get(
                "aspect_ratio",
                0
            ),
            4
        ),

        "extent": round(
            features.get(
                "extent",
                0
            ),
            4
        )
    }

    return result


# ============================================================
# FLASK-FACING FUNCTION
# ============================================================

def analyze_medication():
    """
    Main function that Flask should call.

    This replaces the old mock function.

    Example:

        result = analyze_medication()

    Result:

        {
            "status": "SUCCESS",
            "color": "ORANGE",
            "shape": "CAPSULE",
            "size_mm": 22.40,
            "length_mm": 22.40,
            "width_mm": 11.59,
            ...
        }
    """

    print(
        "Starting medication analysis..."
    )

    # --------------------------------------------------------
    # Capture from FRIEND'S IP CAMERA
    # --------------------------------------------------------

    image = capture_image(
        CAMERA_URL
    )

    if image is None:

        return {
            "status": "ERROR",
            "message": "Could not capture image from IP camera"
        }

    print(
        "Camera image received."
    )

    print(
        "Image size:",
        image.shape
    )

    # --------------------------------------------------------
    # Run CV
    # --------------------------------------------------------

    result = analyze_image(
        image
    )

    # --------------------------------------------------------
    # Print result for server console
    # --------------------------------------------------------

    print(
        "\n========== MEDICATION ANALYSIS =========="
    )

    print(
        "Status:",
        result.get(
            "status"
        )
    )

    if result.get(
        "status"
    ) == "SUCCESS":

        print(
            "Colour:",
            result.get(
                "color"
            )
        )

        print(
            "Shape:",
            result.get(
                "shape"
            )
        )

        print(
            "Length:",
            result.get(
                "length_mm"
            ),
            "mm"
        )

        print(
            "Width:",
            result.get(
                "width_mm"
            ),
            "mm"
        )

        print(
            "Local scale:",
            result.get(
                "pixels_per_mm"
            ),
            "px/mm"
        )

    else:

        print(
            "Message:",
            result.get(
                "message"
            )
        )

    return result

if __name__ == "__main__":
    print("Capturing image from camera...")
    frame = capture_image()
    
    if frame is None:
        print("Failed to capture image. Check camera stream.")
    else:
        results = analyze_image(frame)
        
        if results.get("status") == "SUCCESS":
            print("\n" + "="*30)
            print("   PILL ANALYSIS RESULTS   ")
            print("="*30)
            print(f"Colour : {results.get('color')}")
            print(f"Shape  : {results.get('shape')}")
            print(f"Size   : {results.get('size_mm')} mm")
            print(f"Dims   : {results.get('length_mm')} mm (L) x {results.get('width_mm')} mm (W)")
            print("="*30 + "\n")
        else:
            print(f"Analysis failed: {results.get('message')}")