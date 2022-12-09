import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from gensim.models import Word2Vec, KeyedVectors


TASKS = ['product-pred', 'reactant-pred', 'reagent-pred', 'single-reactant-pred']


def main():
    load_embeddings = False
    for input_format in ['smiles', 'selfies']:
        for token_scheme in ['atom', 'bpe']:
            create_w2v_embeddings(input_format, token_scheme, load_embeddings)
            load_embeddings = True  # training only needed once for my current test (but remove this line later)


def create_w2v_embeddings(input_format, token_scheme, load_embeddings):
    print('Repr: %s , token: %s' % (input_format, token_scheme))
    data_dir = os.path.join('data', TASKS[0], input_format, token_scheme, 'x1')
    if load_embeddings:
        print(' - Loading pre-computed embeddings')
        wv_path = os.path.join(data_dir, 'w2v.wordvectors')
        wv = KeyedVectors.load(wv_path, mmap='r')

    else:
        # Create a dataset of isolated molecules
        print(' - Loading data...')
        with open(os.path.join(data_dir, 'src-train.txt'), 'r') as f_src:
            with open(os.path.join(data_dir, 'tgt-train.txt'), 'r') as f_tgt:
                sentences = []
                for src, tgt in zip(f_src.readlines(), f_tgt.readlines()):
                    molecules = ' . '.join([src.strip(), tgt.strip()])
                    sentences.append(molecules.replace('>', '.').split())

        # Generate model and vocabulary
        print(' - Creating model and building vocabulary...')
        model = Word2Vec(vector_size=256, min_count=1, window=5)
        model.build_vocab(corpus_iterable=sentences)

        # Train the model
        print(' - Training model...')
        model.train(corpus_iterable=sentences,
                    epochs=model.epochs,
                    total_examples=model.corpus_count,
                    total_words=model.corpus_total_words)

        # Save trained embeddings
        wv = model.wv
        wv.save(os.path.join(data_dir, 'w2v.wordvectors'))
        print(' - Training complete.')

    # Write embeddings for each token of this representation and tokenization
    for task in TASKS:
        data_folder_ = os.path.join('data', task, input_format, token_scheme)
        with open(os.path.join(data_folder_, 'w2v-embeddings.txt'), 'w') as f:
            for token in wv.index_to_key:
                vector = wv[token].tolist()
                f.write(token + ' ' + ' '.join(list(map(str, vector))) + '\n')
    print(' - Embeddings saved!')


if __name__ == '__main__':
    main()
