import requests
from bs4 import BeautifulSoup
import csv

# List of plant names
plant_names = [
    "actinidia_chinensis",
    "aegilops_tauschii",
    "amborella_trichopoda",
    "ananas_comosus",
    "arabidopsis_halleri",
    "arabidopsis_lyrata",
    "arabidopsis_thaliana",
    "arabis_alpina",
    "asparagus_officinalis",
    "avena_sativa_ot3098",
    "avena_sativa_sang",
    "beta_vulgaris",
    "brachypodium_distachyon",
    "brassica_juncea",
    "brassica_napus",
    "brassica_oleracea",
    "brassica_rapa",
    "brassica_rapa_ro18",
    "cajanus_cajan",
    "camelina_sativa",
    "cannabis_sativa_female",
    "capsicum_annuum",
    "chara_braunii",
    "chenopodium_quinoa",
    "chlamydomonas_reinhardtii",
    "chondrus_crispus",
    "citrullus_lanatus",
    "citrus_clementina",
    "coffea_canephora",
    "corchorus_capsularis",
    "corylus_avellana",
    "corymbia_citriodora",
    "cucumis_melo",
    "cucumis_sativus",
    "cyanidioschyzon_merolae",
    "cynara_cardunculus",
    "daucus_carota",
    "digitaria_exilis",
    "dioscorea_rotundata",
    "echinochloa_crusgalli",
    "eragrostis_curvula",
    "eragrostis_tef",
    "eucalyptus_grandis",
    "eutrema_salsugineum",
    "ficus_carica",
    "fraxinus_excelsior",
    "galdieria_sulphuraria",
    "glycine_max",
    "gossypium_raimondii",
    "helianthus_annuus",
    "hordeum_vulgare",
    "hordeum_vulgare_goldenpromise",
    "hordeum_vulgare_tritex",
    "ipomoea_triloba",
    "juglans_regia",
    "kalanchoe_fedtschenkoi",
    "lactuca_sativa",
    "leersia_perrieri",
    "lolium_perenne",
    "lupinus_angustifolius",
    "malus_domestica_golden",
    "manihot_esculenta",
    "marchantia_polymorpha",
    "medicago_truncatula",
    "musa_acuminata",
    "nicotiana_attenuata",
    "nymphaea_colorata",
    "olea_europaea",
    "olea_europaea_sylvestris",
    "oryza_barthii",
    "oryza_brachyantha",
    "oryza_glaberrima",
    "oryza_glumipatula",
    "oryza_indica",
    "oryza_longistaminata",
    "oryza_meridionalis",
    "oryza_nivara",
    "oryza_punctata",
    "oryza_rufipogon",
    "oryza_sativa",
    "oryza_sativa_arc",
    "oryza_sativa_azucena",
    "oryza_sativa_chaomeo",
    "oryza_sativa_gobolsailbalam",
    "oryza_sativa_ir64",
    "oryza_sativa_ketannangka",
    "oryza_sativa_khaoyaiguang",
    "oryza_sativa_larhamugad",
    "oryza_sativa_lima",
    "oryza_sativa_liuxu",
    "oryza_sativa_mh63",
    "oryza_sativa_n22",
    "oryza_sativa_natelboro",
    "oryza_sativa_pr106",
    "oryza_sativa_zs97",
    "ostreococcus_lucimarinus",
    "panicum_hallii",
    "panicum_hallii_fil2",
    "papaver_somniferum",
    "phaseolus_vulgaris",
    "physcomitrium_patens",
    "pistacia_vera",
    "pisum_sativum",
    "populus_trichocarpa",
    "prunus_avium",
    "prunus_dulcis",
    "prunus_persica",
    "quercus_lobata",
    "quercus_suber",
    "rosa_chinensis",
    "saccharum_spontaneum",
    "secale_cereale",
    "selaginella_moellendorffii",
    "sesamum_indicum",
    "setaria_italica",
    "setaria_viridis",
    "solanum_lycopersicum",
    "solanum_tuberosum",
    "solanum_tuberosum_rh8903916",
    "sorghum_bicolor",
    "theobroma_cacao",
    "theobroma_cacao_criollo",
    "trifolium_pratense",
    "triticum_aestivum",
    "triticum_aestivum_arinalrfor",
    "triticum_aestivum_cadenza",
    "triticum_aestivum_claire",
    "triticum_aestivum_jagger",
    "triticum_aestivum_julius",
    "triticum_aestivum_kariega",
    "triticum_aestivum_lancer",
    "triticum_aestivum_landmark",
    "triticum_aestivum_mace",
    "triticum_aestivum_mattis",
    "triticum_aestivum_norin61",
    "triticum_aestivum_paragon",
    "triticum_aestivum_refseqv2",
    "triticum_aestivum_renan",
    "triticum_aestivum_robigus",
    "triticum_aestivum_stanley",
    "triticum_aestivum_weebil",
    "triticum_dicoccoides",
    "triticum_spelta",
    "triticum_turgidum",
    "triticum_urartu",
    "vigna_angularis",
    "vigna_radiata",
    "vigna_unguiculata",
    "vitis_vinifera",
    "zea_mays",
]


# Function to extract .gz file links from a given URL
def extract_dna_gz_links(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        gz_links = [url + a["href"] for a in soup.find_all("a") if a["href"].endswith(".dna.toplevel.fa.gz")]
        return gz_links[0]
    else:
        return []
def extract_embl_gz_links(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        gz_links = [url + a["href"] for a in soup.find_all("a") if a["href"].endswith(".57.gff3.gz")]
        return gz_links[0]
    else:
        return []

# Base URL for EMBL .gz files
base_embl_url = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/gff3/{}/"

# Create a CSV file to store the results
with open("ensembl_link.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["name", "gene_link", "annot_link"])

    for plant_name in plant_names:
        embl_url = base_embl_url.format(plant_name)
        embl_link = extract_embl_gz_links(embl_url)

        # Now, get an equal number of DNA links
        dna_url = f"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/{plant_name}/dna/"
        dna_link = extract_dna_gz_links(dna_url)

        csv_writer.writerow([plant_name, dna_link, embl_link])
        # # Add a row for each plant with name, DNA link, and EMBL link separately
        # for dna_link, embl_link in zip(dna_link, embl_link):

print(f"CSV file has been created.")