import math

def roi_pool(feature_map: list, rois: list, output_size: int) -> list:
    """
    Returns a list of pooled grids.
    """
    # Write code here
    pooled_values = []
    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_h = y2 - y1
        roi_w = x2 - x1

        matrix = []
        for i in range(output_size):
            rows = []
            for j in range(output_size):
                w_start = x1 + ((j * roi_w) // output_size)
                w_end = x1 + (((j + 1) * roi_w) // output_size)
                
                h_start = y1 + ((i * roi_h) // output_size)
                h_end = y1 + (((i + 1) * roi_h) // output_size)

                if(h_start == h_end):
                    h_end =  h_start + 1
                if(w_start == w_end):
                    w_end = w_start + 1
                    
                maxi = float('-inf');
                for row in range(h_start, h_end, 1):
                    for col in range(w_start, w_end, 1):
                        maxi = max(maxi, feature_map[row][col])

                rows.append(maxi)

            matrix.append(rows)

        pooled_values.append(matrix)

    return pooled_values