import random

# Read the original file line by line
with open("clean_taka.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Split each line into words using '.' as the delimiter,
# preserving the number of words per line.
lines_words = []
for line in lines:
    # Split on '.' and filter out any empty strings.
    words = [w for w in line.split('.') if w]
    lines_words.append(words)

# Flatten all words from all lines into one list.
all_words = [word for words in lines_words for word in words]

# Shuffle the list of words without replacement.
random.shuffle(all_words)

# Rebuild each line with the same number of words as originally.
shuffled_lines = []
index = 0
for words in lines_words:
    n = len(words)
    # Extract the next n words from the shuffled list.
    line_words = all_words[index:index + n]
    index += n
    # Rejoin words with '.' to reconstruct the line.
    shuffled_line = ".".join(line_words)
    shuffled_lines.append(shuffled_line)

# Combine the lines with newline characters.
shuffled_text = "\n".join(shuffled_lines)

# Write the shuffled text to a new file.
with open("clean_wordshuffled_taka.txt", "w", encoding="utf-8") as f:
    f.write(shuffled_text)

