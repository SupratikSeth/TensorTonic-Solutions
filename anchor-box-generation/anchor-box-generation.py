def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Returns a list of [x1, y1, x2, y2] anchor boxes.
    """
    # Write code here
    stride = image_size / feature_size
    anchor_boxes = []

    for i in range(feature_size):
        for j in range(feature_size):
            c_x = (j + 0.5) * stride
            c_y = (i + 0.5) * stride
            for scale in scales:
                for r in aspect_ratios:
                    w = scale * math.sqrt(r)
                    h = scale / math.sqrt(r)

                    x1 = c_x - w / 2
                    y1 = c_y - h / 2
                    x2 = c_x + w / 2
                    y2 = c_y + h / 2

                    coordinate = [x1, y1, x2, y2]
                    anchor_boxes.append(coordinate)

    return anchor_boxes