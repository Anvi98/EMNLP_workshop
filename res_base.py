
import pandas as pd

# Data for results using CRF plus nomic embeddings and keywords
data_crf_embeddings = {
    'label': ['B-LAW', 'B-VIOLATED BY', 'B-VIOLATED ON', 'B-VIOLATION', 'I-LAW', 'I-VIOLATED BY', 'I-VIOLATED ON', 'I-VIOLATION', 
              'micro avg', 'macro avg', 'weighted avg'],
    'precision_crf_embeddings': [0.85, 0.85, 0.63, 0.78, 0.87, 0.82, 0.78, 0.79, 0.80, 0.80, 0.79],
    'recall_crf_embeddings': [0.68, 0.47, 0.26, 0.61, 0.77, 0.40, 0.26, 0.68, 0.65, 0.52, 0.65],
    'f1-score_crf_embeddings': [0.76, 0.60, 0.37, 0.69, 0.82, 0.54, 0.39, 0.73, 0.71, 0.61, 0.71],
    'support_crf_embeddings': [73, 73, 73, 323, 184, 135, 173, 3700, 4734, 4734, 4734]
}

# Data for results using only tokens with CRF
data_crf_tokens = {
    'label': ['B-LAW', 'B-VIOLATED BY', 'B-VIOLATED ON', 'B-VIOLATION', 'I-LAW', 'I-VIOLATED BY', 'I-VIOLATED ON', 'I-VIOLATION', 
              'micro avg', 'macro avg', 'weighted avg'],
    'precision_crf_tokens': [0.88, 0.86, 0.59, 0.79, 0.93, 0.88, 0.68, 0.78, 0.79, 0.80, 0.79],
    'recall_crf_tokens': [0.75, 0.51, 0.37, 0.61, 0.71, 0.46, 0.43, 0.68, 0.66, 0.56, 0.66],
    'f1-score_crf_tokens': [0.81, 0.64, 0.45, 0.69, 0.80, 0.61, 0.53, 0.73, 0.72, 0.66, 0.71],
    'support_crf_tokens': [71, 71, 71, 352, 183, 149, 152, 3783, 4832, 4832, 4832]
}

# Data for results using POS tagging, embeddings, and CRF
data_pos_tag_embeddings_crf = {
    'label': ['B-LAW', 'B-VIOLATED BY', 'B-VIOLATED ON', 'B-VIOLATION', 'I-LAW', 'I-VIOLATED BY', 'I-VIOLATED ON', 'I-VIOLATION', 
              'micro avg', 'macro avg', 'weighted avg'],
    'precision_pos_tag_embeddings_crf': [0.92, 0.82, 0.52, 0.78, 0.94, 0.88, 0.63, 0.78, 0.78, 0.78, 0.78],
    'recall_pos_tag_embeddings_crf': [0.76, 0.52, 0.39, 0.61, 0.72, 0.48, 0.50, 0.68, 0.66, 0.58, 0.66],
    'f1-score_pos_tag_embeddings_crf': [0.83, 0.64, 0.45, 0.69, 0.81, 0.62, 0.56, 0.73, 0.72, 0.67, 0.72],
    'support_pos_tag_embeddings_crf': [71, 71, 71, 352, 183, 149, 152, 3783, 4832, 4832, 4832]
}

# Create dataframes
df_crf_embeddings = pd.DataFrame(data_crf_embeddings)
df_crf_tokens = pd.DataFrame(data_crf_tokens)
df_pos_tag_embeddings_crf = pd.DataFrame(data_pos_tag_embeddings_crf)

# Merge dataframes
df_results = pd.merge(df_crf_embeddings, df_crf_tokens, on='label', how='outer')
df_results = pd.merge(df_results, df_pos_tag_embeddings_crf, on='label', how='outer')

# Save to CSV
file_path = 'CRF_Results.csv'
df_results.to_csv(file_path, index=False)

print("Results saved to:", file_path)

