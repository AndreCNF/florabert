import requests
from bs4 import BeautifulSoup
import csv
import re

# Base URL for the MaizeGDB website
base_url = "https://download.maizegdb.org/"

# Function to get genomic links for a specific genome assembly
def get_genomic_links(subfolder_url):
    response = requests.get(subfolder_url)
    if response.status_code != 200:
        return None, None

    soup = BeautifulSoup(response.text, "html.parser")
    gene_link = None
    annot_link = None
    f1 = 0 
    f2 = 0 
    # Find links ending with "fa.gz" and "gff3.gz"
    for link in soup.find_all("a"):
        href = link.get("href")
        if href.endswith("fa.gz") and f1 == 0:
            gene_link = subfolder_url + href
            f1 = 1
        elif href.endswith("gff3.gz") and f2 < 2:
            annot_link = subfolder_url + href
            f2 += 1
        if f1 == 1 and f2 == 2:
            break

    return gene_link, annot_link

# Create a CSV file to store the results
csv_filename = "maize_link.csv"
with open(csv_filename, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["name", "gene_link", "annot_link"])

    response = requests.get(base_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        # Find all subfolders that end with "REFERENCE-NAM-1.0" or "REFERENCE-NAM-5.0"
        subfolders = [link.get("href") for link in soup.find_all("a") if link.get("href").endswith(("REFERENCE-NAM-1.0/", "REFERENCE-NAM-5.0/"))]

        for subfolder in subfolders:
            # Retrieve string before "-REFERENCE-NAM" from gene_link
            match = re.search(r"(.+)-REFERENCE-NAM", subfolder)
            if match:
                name = match.group(1)
                # Split name using regex "-"
                name_parts = re.split(r"-", name)
                # Join as first string + lower of second string
                name = name_parts[0] + name_parts[1].lower()

                subfolder_url = base_url + subfolder

                # Get genomic links for this subfolder
                gene_link, annot_link = get_genomic_links(subfolder_url)

                # Write the information to the CSV file
                csv_writer.writerow([name, gene_link, annot_link])

print(f"CSV file '{csv_filename}' has been created.")
