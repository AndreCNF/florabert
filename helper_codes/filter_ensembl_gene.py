import pandas as pd
import urllib.request
import os

# Define a function to get the file size in MB
def get_file_size_mb(url):
    response = urllib.request.urlopen(url)
    file_size_bytes = int(response.headers['Content-Length'])
    file_size_mb = file_size_bytes / (1024 * 1024)
    return file_size_mb

# Read the CSV file into a DataFrame
df = pd.read_csv('ensembl_link.csv')

# Create a new DataFrame to store the filtered data
filtered_df = pd.DataFrame(columns=df.columns)

# Iterate through each row in the original DataFrame
for index, row in df.iterrows():
    gene_link = row['gene_link']
    
    # Get the file size for the current link
    file_size_mb = get_file_size_mb(gene_link)
    
    # Check if the file size is less than or equal to 650 MB
    if file_size_mb <= 112:
        filtered_df = pd.concat([filtered_df, row.to_frame().T], ignore_index=True)

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('filtered_ensembl.csv', index=False)
