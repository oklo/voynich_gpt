#### Voynich manuscript analysis

ChatGPT discussion at Voynich Transformer Modeling

Two research threads are charted here. The first, voynich_analysis.ipynb, is an N-gram focused assessment of the downloaded manuscript text, the second is a comparison of tiny 150K-parameter GPT-architecutre autoregressive language character-level models including the Voynich text and equivalent-length English, French, Tagalog and Hawaiian texts (where the latter two are used due to their low language entropies).

The setup uses a python virtual environment (rather than a conda env) to access torch and additional packages. Activate the environment at the bash prompt via:

%source /Users/gl396/voynich_env/bin/activate

voynich_env is additionally available as a kernel for the notebooks.

The data/ directory has the initial processing workflow. The raw files, clean_{filename}.txt, are obtained via copy-paste from the Internet, with the Voynich Manuscript accessed using 

!curl -O https://www.voynich.nu/data/IT2a-n.txt

which is processed by voynich_analysis.ipynb to data/voynich_char/clean_taka.txt

%python data/voynich_char/prepare.py

accesses the text file and generates training and validation data (train.bin, val.bin, along with a metadata descriptor file, meta.pkl). Tokenization here is on a per-character basis.

%python data/voynich_char/shuffle.py

shuffles the text at the word level, and places clean_wordshuffled_taka.txt in its own subdirectory within data, where prepare.py is used to make the character-level training data. Similar workflow applies to the other texts.

Train the model in the bash shell:
%python train.py config/train_voynich_char.py 

where the config contains an architecture that generates a 150K parameter model. See the discussion with ChatGPT on flops utilization, etc. The model checkpoint and training log is at out-voynich-char/ . The model trains in several minutes on the Intel CPU.

Obtain inference via

%python sample.py --out_dir=out-voynich-char

The quality of the inferences can be assessed by looking at the analogous-sized model trained on the English text.