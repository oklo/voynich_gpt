# train a miniature character-level small GPT

out_dir = 'out-voynich-wordscramble-char'
eval_interval = 120 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 1 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like

dataset = 'voynich_wordscramble_char'
gradient_accumulation_steps = 1
batch_size = 10
block_size = 32 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 3
n_head = 4
n_embd = 64
dropout = 0.01

learning_rate = 2.5e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 50 # not super necessary potentially

device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model

