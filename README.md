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
- [`2-feature-visualization/`](https://github.com/gurveervirk/florabert/tree/master/scripts/2-feature-visualization)
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

### Personal Updates on Forked Repo: (Removed due to low google drive space)

**First module has been completed. All data / outputs are under [`data`](https://github.com/gurveervirk/florabert/tree/main/data) or [`models`](https://github.com/gurveervirk/florabert/tree/main/models). Moving to Second Module. The following steps were essential for this [script](https://github.com/gurveervirk/florabert/blob/main/scripts/0-data-loading-processing/04-process-genex-nam.py).**

The following updates have been done using python scripts under [`3-RNAseq-quantification/`](https://github.com/gurveervirk/florabert/tree/master/scripts/3-RNAseq-quantification):

- The scripts under this module requires a lot of resources and time (patience). We opted to use the Bioinformatics website [Galaxy](https://usegalaxy.org/). This provides every user 250GB storage and allows the ability to use a number of very useful and important bioinformatics tools. 

- The scripts under the module dealt with 26 NAM lines / cultivars of Maize. We replicated the entire process under this module in this website, with some minor changes (not in output). The first step was to get all the runs corresponding to each cultivar and unique organsim part for each, to avoid repitition. 

- This was achieved by getting the base data from [EBI](https://www.ebi.ac.uk/) and searching for the 2 Bioprojects mentioned in the supplementary material (under [`research_papers`](https://github.com/gurveervirk/florabert/tree/main/research_papers)). This data was then used alongside the [`helper_codes`](https://github.com/gurveervirk/florabert/tree/main/helper_codes) scripts to get the file [`unique_orgs_runs.tsv`](https://github.com/gurveervirk/florabert/blob/main/helper_files/unique_orgs_runs.tsv). This file contains the runs corresponding to unique organism parts of each cultivar.

- A workflow was then created / implemented / configured (the base workflow was created by user vasquex11 on the mentioned website) to align with the scripts. The runs were first uploaded per cultivar to the website (after logging in) in txt format, one per line. Next, fasterq-dump tool was used with --split-files option selected to get the fastq files corresponding to the runs.

- The created workflow [`FloraBERT Test (Trimmomatic + HISAT2 + featureCounts)`](https://usegalaxy.org/u/gurveer05/w/copy-of-module-72-part-1-trimmomatic--hisat2--featurecounts-shared-by-user-vasquex11) was used to perform all the actions mentioned in the module. The final output are the featureCounts files corresponding to each run ( extending to unique organsim part of cultivars ). The steps are self-explanatory (using the research papers).

**The full train-test data for pretraining is available at [`Kaggle Dataset For Pretraining`](https://www.kaggle.com/datasets/gsv001100/dataset-for-updation/versions/7) and for finetuning at [`Kaggle Dataset For Finetuning`](https://www.kaggle.com/datasets/gsv001100/dataset-for-updation/versions/3)**

**Some observations**:

- using different transformations to handle the highly right skewed TPM values (during finetuning stage):
  - natural log transformation gave an mse of 2.4
  - boxcox transformation gave an mse of 12.9
  - log10 transformation gave an mse of 0.4441!?!?!?
  - log10 also improved r2 by 0.01 (from 0.077 to 0.087) which is considerable here (no other changes)
  - this means that handling the highly right skewed is pivotal here for model performance
- TPU usage on pytorch is a herculean task for our use-case
- custom finetuning script may have to be tweaked
- python dependencies are problematic at times
- DNABERT-1 makes use of k-mers meaning the tokenizer files that are made are basically 4^k permutations ('A', 'T', 'G', 'C'), not great (now) for tasks dealing with DNA
- DNABERT-2 is an obvious improvement as it makes use of actual tokenization, preventing overlapping and better identification of important parts of DNA seqeunces
- Byte-level-BPE-tokenizer makes use of 256 basic (byte-level) tokens; taking this into consideration, new vocab_size is set to 5256
- padding HAS GOT TO BE EQUAL TO true OR "max_length" WHEN TRAINING ON TPU (as a parameter for tokenizer; used in preprocess_fn in dataio.py in this proj)
- torch, torch_xla, torch_xla.core.xla_model (as xm) have to be imported to make sure Training (Trainer) works properly on TPU (tested on kaggle)
