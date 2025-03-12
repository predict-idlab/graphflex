from .base import PostProcessing
import numpy as np

class DropDuplicateEdges(PostProcessing):
    def reduce(self, matrix, column_dct):
        _, unique_indices = np.unique(matrix.T, axis=0, return_index=True)
        unique_indices.sort()
        filtered_matrix = matrix[:, unique_indices]
        inv_column_dict = {column_dct[key]: key for key in column_dct}
        filtered_column_dict = {inv_column_dict[unique_indices[i]]: i for i in range(len(unique_indices))}
        return filtered_matrix, filtered_column_dict

class NonUniqueFeatureFilter(PostProcessing):
    def __init__(self, threshold_value=1):
        self.threshold_value = threshold_value

    def reduce(self, matrix, column_dct):
        num_rows = matrix.shape[0]

        threshold_value_zeros = self.threshold_value
        threshold_value_ones = num_rows - self.threshold_value

        # Reverse dictionary mapping
        inv_column_dct = {v: k for k, v in column_dct.items()}

        # Vectorized count of nonzero elements
        nonzero_counts = np.count_nonzero(matrix, axis=0)
        ones_counts = np.count_nonzero(matrix == 1, axis=0)

        # Boolean mask for valid columns
        mask = (nonzero_counts >= threshold_value_zeros) & (ones_counts <= threshold_value_ones)

        # Ensure columns with "(real_date)" or "(length)" in names are always kept
        column_names = np.array([inv_column_dct[i] for i in range(matrix.shape[1])])
        special_columns = (np.char.find(column_names, "(real_date)") != -1) | (
                    np.char.find(column_names, "(length)") != -1)

        # Final selection mask
        final_mask = mask | special_columns

        # Filter matrix and update column dictionary
        filtered_matrix = matrix[:, final_mask]
        valid_indices = np.where(final_mask)[0]
        filtered_column_dct = {inv_column_dct[idx]: i for i, idx in enumerate(valid_indices)}

        ## duplicate drop
        filtered_matrix, filtered_column_dct = DropDuplicateEdges().reduce(filtered_matrix, filtered_column_dct)
        return filtered_matrix, filtered_column_dct

