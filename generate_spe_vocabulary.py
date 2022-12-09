import os
import codecs
import pandas as pd
from SmilesPE.learner import *
from SmilesPE.tokenizer import *

# TODO: selfiesify the zink dataset and create spe-tokenizer for zink-selfies

# Load zink in-data file and prepare vocab out-data file
inpath = os.path.join('data', 'zink', 'allsmiles_filtered.txt')
infile = pd.read_csv(inpath).smi.tolist()
outpath = os.path.join('data', 'zink', 'spe_vocab.txt')
outfile = codecs.open(outpath, 'w')

# Train spe (smiles-bpe) tokenizer on the zink data set
learn_SPE(infile=infile,
          outfile=outfile,
          num_symbols=30000,
          min_frequency=2000,
          augmentation=1,
          verbose=True,
          total_symbols=True)
