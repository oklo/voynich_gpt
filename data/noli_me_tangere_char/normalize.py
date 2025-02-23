import re
import unicodedata

def normalize_text(text):
    # Remove bracketed annotations, e.g. [49], [50], etc.
    text = re.sub(r'\[[^\]]*\]', '', text)
    # Normalize to NFKD form to decompose accented characters
    normalized = unicodedata.normalize('NFKD', text)
    # Remove combining diacritical marks (e.g., accents)
    ascii_text = ''.join(c for c in normalized if not unicodedata.combining(c))
    return ascii_text

def main():
    input_filename = 'noli_me_tangere.txt'
    output_filename = 'clean_noli_me_tangere.txt'
    
    # Read the input file
    with open(input_filename, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Normalize and clean the text
    cleaned_text = normalize_text(raw_text)
    
    # Write the cleaned text to the output file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"Cleaned text written to {output_filename}")

if __name__ == '__main__':
    main()

