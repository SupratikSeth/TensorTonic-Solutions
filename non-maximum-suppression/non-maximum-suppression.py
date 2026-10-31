def nms(boxes: list, scores: list, iou_threshold: float) -> list:
    """
    Returns a list of retained indices.
    """
    # Write code here
    if not boxes:
        return []
    
    triplet = [(box, score, idx) for idx, (box, score) in enumerate(zip(boxes, scores))]
    
    triplet_sorted = sorted(triplet, key=lambda box: box[1], reverse=True)

    
    final_list = []
    
    while triplet_sorted:
        confident_box, score, idx = triplet_sorted[0]
        final_list.append(idx)
        
        remaining = []
        for i in range(1, len(triplet_sorted), 1):
            x1_B, y1_B, x2_B, y2_B = triplet_sorted[i][0]
            x1_A, y1_A, x2_A, y2_A = confident_box
    
            x1_int = max(x1_A, x1_B)
            y1_int = max(y1_A, y1_B)
            x2_int = min(x2_A, x2_B)
            y2_int = min(y2_A, y2_B)
    
            area_intersection = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
            area_A = (x2_A - x1_A) * (y2_A -y1_A)
            area_B = (x2_B - x1_B) * (y2_B - y1_B)
            area_union = area_A + area_B - area_intersection
    
            iou = area_intersection / area_union
    
            if iou < iou_threshold:
                remaining.append(triplet_sorted[i])

        triplet_sorted = remaining

    return final_list