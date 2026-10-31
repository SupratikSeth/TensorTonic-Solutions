def iou(box_a: list, box_b: list) -> float:
    """
    Returns IoU as a float.
    """
    # Write code here
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    x1_int = max(x1_a, x1_b)
    y1_int = max(y1_a, y1_b)
    x2_int = min(x2_a, x2_b)
    y2_int = min(y2_a, y2_b)

    area_intersection = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    area_union = area_a + area_b - area_intersection

    iou = area_intersection / area_union

    return iou