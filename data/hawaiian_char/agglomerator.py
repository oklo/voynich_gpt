import glob
import re

# Define regex patterns:
# Pattern to remove roman numeral tokens (assumes uppercase roman numerals)
roman_pattern = re.compile(r'\b[MCDXLIV]+\b')
# Pattern to remove square brackets and any text inside them
bracket_pattern = re.compile(r'\[[^\]]*\]')

# Glob the files whose names start with "haw" and end with ".txt"
file_list = glob.glob("textfiles/haw*.txt")

all_text = []

for filename in file_list:
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Skip the first two lines of each file
    content = "".join(lines[2:])
    # Remove roman numeral tokens
    content = roman_pattern.sub("", content)
    # Remove any square brackets and text inside them
    content = bracket_pattern.sub("", content)
    # Optionally, you can strip extra whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    all_text.append(content)

# Concatenate all processed texts, separating files with a newline
final_text = "\n".join(all_text)

# Write the concatenated result to an output file
with open("concatenated.txt", "w", encoding="utf-8") as out_file:
    out_file.write(final_text)

print("Concatenated text written to concatenated.txt")

