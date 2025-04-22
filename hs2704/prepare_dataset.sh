#!/bin/bash
set -e

DATASET_REPO=https://github.com/Rishit-dagli/CPPE-Dataset.git
TARGET_DIR=${TARGET_DIR:-${HOME}/CPPE-Dataset}
[ -d ${TARGET_DIR} ] && echo "Pre-existing directory at ${TARGET_DIR}. Remove and rerun" && exit 1

cd /tmp
[ -d CPPE-Dataset ] && echo "Pre-existing directory at /tmp/CPPE-Dataset. Remove and rerun" && exit 1

set -x
git clone ${DATASET_REPO} CPPE-Dataset
cd /tmp/CPPE-Dataset

## Note that in some environments, the gdown utility is installed in $HOME/.local/bin
## With supporting scripts and files in $HOME/.local/lib/python3.10/site-packages
## If so, pls run following and retry the download_data script and move the downloaded dataset to
## desired target directory manually.
## 
##     export PATH=$PATH:${HOME}/.local/bin

bash tools/download_data.sh
mv /tmp/CPPE-Dataset ${TARGET_DIR}
chmod -R 777 ${TARGET_DIR}

echo Script completed.
