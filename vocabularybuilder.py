# vocabularybuilder.py

import os
import pickle
import nltk
import spacy
from collections import Counter
from itertools import islice

def build_vocabulary(file_path, language_model, min_freq=2, max_vocab_size=100000, num_lines=1600000):
    nlp = spacy.load(language_model)
    tokens = []
    
    # Read file line by line, tokenize and lowercase and remove non alphabetic tokens and punctuation
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read only first num_lines lines
        lines = file if num_lines is None else islice(file, num_lines)

        for line in lines:
            doc = nlp(line.strip())
            tokens.extend([token.text.lower() for token in doc if token.text.isalpha() or token.is_punct])


    vocabulary = Counter(tokens)
    # Filter by minimum frequency
    vocabulary = {word: count for word, count in vocabulary.items() if count >= min_freq}
    # Limit vocabulary size
    if max_vocab_size:
        vocabulary = dict(Counter(vocabulary).most_common(max_vocab_size))

    # Create a word-to-index dictionary and add special tokens first
    word_to_index = {
        '<PAD>': 0, 
        '<SOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3
    }

    # Add other words starting indices from 4
    word_to_index.update({word: idx + 4 for idx, word in enumerate(vocabulary.keys())})

    return word_to_index

def save_vocabulary(vocabulary, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(vocabulary, file)
    print(f"Vocabulary saved to {file_path}")

    # Optionally, save the vocabulary to a text file for easy reading
    text_file_path = file_path.replace('.pkl', '.txt')
    with open(text_file_path, 'w', encoding='utf-8') as file:
        for word, index in vocabulary.items():
            file.write(f"{word}: {index}\n")
    print(f"Vocabulary contents saved to {text_file_path}")

def main(file_en, file_sv):
    vocab_en = build_vocabulary(file_en, "en_core_web_sm")
    vocab_sv = build_vocabulary(file_sv, "sv_core_news_sm")
    
    save_vocabulary(vocab_en, 'vocabulary_en_2.pkl')
    save_vocabulary(vocab_sv, 'vocabulary_sv_2.pkl')

if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])  # Provide file paths to English and Swedish text files as arguments