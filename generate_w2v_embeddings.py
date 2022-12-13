import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from gensim.models import Word2Vec
from itertools import product as combine


TASKS = ['product-pred', 'reactant-pred', 'reagent-pred']
INPUT_FORMATS = ['smiles', 'selfies']
TOKEN_SCHEMES = ['atom', 'spe']


def main():
    for input_format, token_scheme in combine(INPUT_FORMATS, TOKEN_SCHEMES):
        create_w2v_embeddings(input_format, token_scheme)


def create_w2v_embeddings(input_format, token_scheme):
    print('Repr: %s , token: %s' % (input_format, token_scheme))
    data_dir_in = os.path.join(
        'data', TASKS[0], input_format, token_scheme, 'x1')

    # Consider reactions as sets of dot-separated molecules
    # The goal is to share src and tgt embeddings for any task
    print(' - Loading data...')
    with open(os.path.join(data_dir_in, 'src-train.txt'), 'r') as f_src,\
         open(os.path.join(data_dir_in, 'tgt-train.txt'), 'r') as f_tgt:
        sentences = []
        for src, tgt in zip(f_src.readlines(), f_tgt.readlines()):
            molecules = ' . '.join([src.strip(), tgt.strip()])
            sentences.append(molecules.split())

    # Generate model and vocabulary
    print(' - Creating model and building vocabulary...')
    model = Word2Vec(vector_size=256, min_count=1, window=5)
    model.build_vocab(corpus_iterable=sentences)  # should check clash with omt

    # Train the model and save embbeding vectors
    print(' - Training model...')
    model.train(corpus_iterable=sentences,
                epochs=model.epochs,
                total_examples=model.corpus_count,
                total_words=model.corpus_total_words)
    wv = model.wv
    print(' - Training complete.')

    # Write embeddings for each token of this representation and tokenization
    for task in TASKS:
        data_dir_out = data_dir_in.replace(TASKS[0], task)
        wv.save(os.path.join(data_dir_out, 'w2v.wordvectors'))
        with open(os.path.join(data_dir_out, 'w2v-embeddings.txt'), 'w') as f:
            for token in wv.index_to_key:
                vector = wv[token].tolist()
                f.write(token + ' ' + ' '.join(list(map(str, vector))) + '\n')
    print(' - Embeddings saved!')


if __name__ == '__main__':
    main()
