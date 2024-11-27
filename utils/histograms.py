import numpy as np
from numba import njit

@njit
def rebin_histogram(counts, bin_edges, new_bin_edges):
    new_counts = np.zeros(len(new_bin_edges) - 1)

    for i in range(len(new_bin_edges) - 1):
        new_bin_start, new_bin_end = new_bin_edges[i], new_bin_edges[i+1]
        
        # Loop over each fine bin
        for j in range(len(bin_edges) - 1):
            fine_bin_start, fine_bin_end = bin_edges[j], bin_edges[j+1]

            if fine_bin_end <= new_bin_start:
                continue
            if fine_bin_start >= new_bin_end:
                break
            
            # Check for overlap between fine bin and new bin
            overlap_start = max(new_bin_start, fine_bin_start)
            overlap_end = min(new_bin_end, fine_bin_end)
            
            if overlap_start < overlap_end:
                # Calculate the overlap width
                overlap_width = overlap_end - overlap_start
                fine_bin_width = fine_bin_end - fine_bin_start
                
                # Proportion of the fine bin's count that goes into the new bin
                contribution = (overlap_width / fine_bin_width) * counts[j]
                
                # Add the contribution to the new bin's count
                new_counts[i] += contribution
    
    return new_counts

@njit
def rebin_histogram2d(counts, x_bin_edges, y_bin_edges, new_x_bin_edges, new_y_bin_edges):
    new_counts = np.zeros((len(new_x_bin_edges) - 1, len(new_y_bin_edges) - 1))

    for i in range(len(new_x_bin_edges) - 1):
        new_x_bin_start, new_x_bin_end = new_x_bin_edges[i], new_x_bin_edges[i+1]
        
        for j in range(len(new_y_bin_edges) - 1):
            new_y_bin_start, new_y_bin_end = new_y_bin_edges[j], new_y_bin_edges[j+1]

            for x in range(len(x_bin_edges) - 1):
                x_bin_start, x_bin_end = x_bin_edges[x], x_bin_edges[x+1]

                if x_bin_end <= new_x_bin_start:
                    continue
                if x_bin_start >= new_x_bin_end:
                    break

                for y in range(len(y_bin_edges) - 1):
                    y_bin_start, y_bin_end = y_bin_edges[y], y_bin_edges[y+1]

                    if y_bin_end <= new_y_bin_start:
                        continue
                    if y_bin_start >= new_y_bin_end:
                        break

                    overlap_x_start = max(new_x_bin_start, x_bin_start)
                    overlap_x_end = min(new_x_bin_end, x_bin_end)
                    overlap_y_start = max(new_y_bin_start, y_bin_start)
                    overlap_y_end = min(new_y_bin_end, y_bin_end)

                    if overlap_x_start < overlap_x_end and overlap_y_start < overlap_y_end:
                        overlap_x_width = overlap_x_end - overlap_x_start
                        x_bin_width = x_bin_end - x_bin_start
                        overlap_y_width = overlap_y_end - overlap_y_start
                        y_bin_width = y_bin_end - y_bin_start

                        contribution = (overlap_x_width / x_bin_width) * (overlap_y_width / y_bin_width) * counts[x, y]

                        new_counts[i, j] += contribution
    
    return new_counts

def centers_to_edges(centers):
    return np.concatenate((centers - (centers[1] - centers[0]) / 2, [centers[-1] + (centers[1] - centers[0]) / 2]))