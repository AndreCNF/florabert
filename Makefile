.PHONY: genomics_tools work

# ---------- GLOBALS ---------- #
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SHELL := /bin/bash

# all : python_requirements genomics_tools

# python_requirements: requirements.txt
# 	# virtualenv env \
# 	# && source ./env/bin/activate \
# 	# && pip install -r requirements.txt
# 	# remove if virtualenv is being used
# 	pip install -r requirements.txt

genomics_tools: samtools bedtools

# Installing samtools
samtools :
	curl -L https://github.com/samtools/samtools/releases/download/1.19.2/samtools-1.19.2.tar.bz2 \
		| tar -xj \
		&& cd samtools-1.19.2 \
		&& ./configure --prefix=$(PROJECT_DIR) --without-curses \
		&& make \
		&& make install \
		&& cd $(PROJECT_DIR) \
		&& ln -s $(PROJECT_DIR)/samtools-1.19.2/samtools $(PROJECT_DIR)/libs/samtools

# Installing bedtools
bedtools :
# linux option:
	# curl -L https://github.com/arq5x/bedtools2/releases/download/v2.31.1/bedtools-2.31.1.tar.gz \
	# 	| tar zxv \
	# 	&& cd bedtools2 \
	# 	&& make \
	# 	&& cd $(PROJECT_DIR) \
	# 	&& ln -s $(PROJECT_DIR)/bedtools2/bin/bedtools $(PROJECT_DIR)/libs/bedtools
# mac option:
	# brew install bedtools
	ln -s /opt/homebrew/bin/bedtools $(PROJECT_DIR)/libs/bedtools