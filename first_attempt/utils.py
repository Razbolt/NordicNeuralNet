# utils.py

import argparse
import yaml
from nltk.translate.bleu_score import corpus_bleu

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process settings from a YAML file.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration file')
    return parser.parse_args()


def read_settings(config_path):
    with open(config_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings

def calculate_bleu_score(references, hypotheses):
     # Print example data structures
    #print("Sample References:", references[:2])  # Show the first two sets of references
    #print("Sample Hypotheses:", hypotheses[:2])  # Show the first two hypotheses

    # Check the full structure of the first complete example if needed
    if references and hypotheses:  # Ensure there is data to avoid index errors
        print("Detailed view of the first reference set:", references[0])
        print("Detailed view of the first hypothesis:", hypotheses[0])

    # Proceed to calculate the BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score