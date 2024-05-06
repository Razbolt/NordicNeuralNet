import re

def clean_text(text):
    # Replace numbers with a placeholder
    text = re.sub(r'\d+', '<NUM>', text)
    
    # Remove unwanted special characters, but keep Swedish letters and common punctuation
    text = re.sub(r'[^\w\s,.?!åäöÅÄÖ]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()

    return text

def process_file(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        for line in infile:
            cleaned_line = clean_text(line)
            outfile.write(cleaned_line + '\n')

# File paths
input_filepath = 'combined_data.txt'
output_filepath = 'cleaned_combined_data.txt'

# Process the file
process_file(input_filepath, output_filepath)
print("Finished processing the text.")