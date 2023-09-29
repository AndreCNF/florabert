# Bringing BERT to the field: Transformer models for gene expression prediction in maize

**Authors: Benjamin Levy, Shuying Ni, Zihao Xu, Liyang Zhao**  
Predicting gene expression levels from upstream promoter regions using deep learning. Collaboration between IACS and Inari.

---

## Directory Setup

**`scripts/`: directory for production code**

- [`0-data-loading-processing/`](https://github.com/gurveervirk/florabert/tree/master/scripts/0-data-loading-processing):
  - [`01-gene-expression.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/01-gene-expression.py): downloads and processes gene expression data and saves into "B73_genex.txt".
  - [`02-download-process-db-data.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/02-download-process-db-data.py): downloads and processes gene sequences from a specified database: 'Ensembl', 'Maize', 'Maize_addition', 'Refseq'
  - [`03-combine-databases.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/03-combine-databases.py): combines all the downloaded sequences within all the databases
  - [`04a-merge-genex-maize_seq.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/04a-merge-genex-maize_seq.py):
  - [`04b-merge-genex-b73.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/04b-merge-genex-b73.py):
  - [`05a-cluster-maize_seq.sh`](scripts/0-data-loading-processing/05a-cluster-maize_seq.sh): clusters the promoter sequences into groups with up to 80% sequence identity, which may be interpreted as paralogs
  - [`05b-train-test-split.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/05-train-test-split.py): divides the promoter sequences into train and test sets, avoiding a set of pairs that indicate close relations ("paralogs")
  - [`06_transformer_preparation.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/06_transformer_preparation.py):
  - [`07_train_tokenizer.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/07_train_tokenizer.py): training byte-level BPE for RoBERTa model
- [`1-modeling/`](https://github.com/gurveervirk/florabert/tree/master/scripts/1-modeling)
  - [`pretrain.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/1-modeling/pretrain.py): training the FLORABERT base using a masked language modeling task. Type `python scripts/1-modeling/pretrain.py --help` to see command line options, including choice of dataset and whether to warmstart from a partially trained model. Note: not all options will be used by this script.
  - [`finetune.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/1-modeling/finetune.py): training the FLORABERT regression model (including newly initialized regression head) on multitask regression for gene expression in all 10 tissues. Type `python scripts/1-modeling/finetune.py --help` to see command line options; mainly for specifying data inputs and output directory for saving model weights.
  - [`evaluate.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/1-modeling/evaluate.py): computing metrics for the trained FLORABERT model
- [`2-feature-visualization/](https://github.com/gurveervirk/florabert/tree/master/scripts/2-feature-visualization)`
  - [`embedding_vis.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/2-feature-visualization/embedding_vis.py): computing a sample of BERT embeddings for the testing data and saving to a tensorboard log. Can specify how many embeddings to sample with `--num-embeddings XX` where `XX` is the number of embeddings (must be integer).

**`module/`: directory for our customized modules**

- [`module/`](https://github.com/gurveervirk/florabert/tree/master/module/florabert): our main module named `florabert` that packages customized functions
  - [`config.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/config.py): project-wide configuration settings and absolute paths to important directories/files
  - [`dataio.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/dataio.py): utilities for performing I/O operations (reading and writing to/from files)
  - [`gene_db_io.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/gene_db_io.py): helper functions to download and process gene sequences
  - [`metrics.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/metrics.py): functions for evaluating models
  - [`nlp.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/nlp.py): custom classes and functions for working with text/sequences
  - [`training.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/training.py): helper functions that make it easier to train models in PyTorch and with Huggingface's Trainer API, as well as custom optimizers and schedulers
  - [`transformers.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/transformers.py): implementation of RoBERTa model with mean-pooling of final token embeddings, as well as functions for loading and working with Huggingface's transformers library
  - [`utils.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/utils.py): General-purpose functions and code
  - [`visualization.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/visualization.py): helper functions to perform random k-mer flip during data processing and make model prediction

### Pretrained models

If you wish to experiment with our pre-trained FLORABERT models, you can find the saved PyTorch models and the Huggingface tokenizer files [here](https://drive.google.com/drive/folders/1qHwRfXxPVC1j2GcZ-wFOT3BmTmHRr_it?usp=sharing)

**Contents**:

- `byte-level-bpe-tokenizer`: Files expected by a Huggingface `transformers.PretrainedTokenizer`
  - `merges.txt`
  - `vocab.txt`
- transformer: Both language models can instantiate any RoBERTa model from Huggingface's `transformers` library. The prediction model should instantiate our custom `RobertaForSequenceClassificationMeanPool` model class
  1. `language-model`: Trained on all plant promoter sequences
  2. `language-model-finetuned`: Further trained on just maize promoter sequences
  3. `prediction-model`: Fine-tuned on the multitask regression problem

---

### Personal Updates on Forked Repo:

The following updates have been done using python scripts under [`0-data-loading-processing/`](https://github.com/gurveervirk/florabert/tree/master/scripts/0-data-loading-processing)

**Data from step 2 using Refseq links**:

- Install zip file from [here](https://drive.google.com/file/d/1-0V8grOh1zh4-4EisXy_fxqQeAHuySdu/view?usp=drive_link) --> (contains data folder after step 2 using refseq links)
- unzip and add to florabert, if needed
- further testing required

**Data from step 2 using Ensembl links**:

- Install zip file from [here](https://drive.google.com/file/d/11_ZOm3l7sakyAwxhiEGYsaj3if1YK8Cu/view?usp=drive_link) --> (contains data folder after step 2 using ensembl links)
- unzip and add to florabert, if needed
- further testing required

**Data from step 2 using Maize NAM links**:

- Install zip file from [here](https://drive.google.com/file/d/1xK8w0h_ttmMV0TBDSStKYt_DHeYh0F3q/view?usp=sharing) --> (contains data folder after step 2 using maize_nam (from MaizeGDB FTP; only NAM lines) links)
- unzip and add to florabert, if needed
- further testing required

Previous 3 steps, if used together, need to be merged to get the correct folder structure and then run 3rd file under data-loading folder

**Data from step 3**:

- Install zip file from [here](https://drive.google.com/file/d/1hNzlP4xHU0fLT5GxHUWKYyoXthsEcXH7/view?usp=sharing) --> (contains data folder after step 3)
- unzip and add to florabert, if needed
- further testing required

**Data from step 5**:

- Install file from [here](https://drive.google.com/file/d/1iTrmHDawZpi33Cv5GsgYei4H5xz06yMs/view?usp=sharing) --> (contains data folder after step 5)
- add to florabert, if needed
- further testing required

**Data from step 6**:

- Install file from [here](https://drive.google.com/file/d/1c-JXUcC4mnepp_SV5O0Rheb3qbN3LaZf/view?usp=sharing) --> (contains data folder after step 6)
- add to florabert, if needed
- further testing required