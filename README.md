
# Summarization of Scientific Articles with T5

Welcome to the Summarization of Scientific Articles with T5 project! This project focuses on summarizing scientific articles using the T5 model.

## Introduction

Summarization involves condensing text to its key points. In this project, we leverage T5 to summarize scientific articles using a dataset of articles and their summaries.

## Dataset

For this project, we will use a custom dataset of scientific articles and their summaries. You can create your own dataset and place it in the `data/scientific_articles.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets
- Pandas

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/asimsultan/summarization_scientific_articles_t5.git
cd summarization_scientific_articles_t5

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes scientific articles and their summaries. Place these files in the data/ directory.
# The data should be in a CSV file with two columns: article and summary.

# To fine-tune the T5 model for summarization, run the following command:
python scripts/train.py --data_path data/scientific_articles.csv

# To evaluate the performance of the fine-tuned model, run:
python scripts/evaluate.py --model_path models/ --data_path data/scientific_articles.csv
