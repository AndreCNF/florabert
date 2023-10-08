import requests
from bs4 import BeautifulSoup
import csv
import os

# Base URL for the plant genomes
base_url = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/plant/"

# Function to get genomic links for a specific genome assembly
def get_genomic_links(genome_assembly_url, genome_assembly_name):
    # response = requests.get(genome_assembly_url)
    # if response.status_code != 200:
    #     return None, None

    # soup = BeautifulSoup(response.text, "html.parser")
    # gene_link = None
    # annot_link = None

    # # Find links ending with "_genomic.fna.gz" and "_genomic.gff.gz"
    # for link in soup.find_all("a"):
    #     href = link.get("href")
    #     if href.endswith("_genomic.fna.gz"):
    #         gene_link = genome_assembly_url + "/" + href
    #     elif href.endswith("_genomic.gff.gz"):
    #         annot_link = genome_assembly_url + "/" + href

    return genome_assembly_url + "/" + genome_assembly_name + "_genomic.fna.gz", genome_assembly_url + "/" + genome_assembly_name + "_genomic.gff.gz"

# Create a CSV file to store the results
i = 0
csv_filename = "refseq_link.csv"
with open(csv_filename, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["name", "gene_link", "annot_link"])

    response = requests.get(base_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a")[1:-3]:
            href = link.get("href")
            # Check if the link is a directory (ends with "/") and not a file
            if href.endswith("/") and not href.endswith(".gz"):
                genome_assembly_name = href.rstrip("/")  # Remove trailing slash from the URL
                genome_assembly_url = f"{base_url}{genome_assembly_name}"

                # Check if the genome assembly folder has an "all_assembly_versions" subfolder
                all_versions_url = f"{genome_assembly_url}/all_assembly_versions"
                all_versions_response = requests.get(all_versions_url)

                if all_versions_response.status_code == 200:
                    all_versions_soup = BeautifulSoup(all_versions_response.text, "html.parser")
                    for version_link in all_versions_soup.find_all("a")[1:]:
                        version_href = version_link.get("href")
                        # Check if the link is a directory and starts with "GCF"
                        if version_href.endswith("/") and version_href.startswith("GCF"):
                            version_name = version_href.rstrip("/")  # Remove trailing slash from the URL
                            version_url = f"{all_versions_url}/{version_name}"

                            # Get genomic links for this version
                            gene_link, annot_link = get_genomic_links(version_url, version_name)
                            # print(gene_link, annot_link)

                            # Write the information to the CSV file
                            csv_writer.writerow([version_name, gene_link, annot_link])
                            print(href, i)
                            i += 1

print(f"CSV file '{csv_filename}' has been created.")
