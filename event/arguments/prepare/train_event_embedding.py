import os
from gensim.models.word2vec import Word2Vec
import glob
from smart_open import open


def train_event_vectors(input_pattern, vector_out_base, window_size, min_count=5):
    class Sentences:
        def __init__(self, pattern):
            self.input_pattern = pattern

        def __iter__(self):
            for sent_file in glob.glob(self.input_pattern):
                if not os.path.exists(sent_file):
                    print(f"Warning: provided path not exists: {sent_file}")
                with open(sent_file) as doc:
                    print("Processing ", sent_file)
                    for line in doc:
                        yield line.split()

    model = Word2Vec(
        Sentences(input_pattern),
        workers=10,
        size=300,
        window=window_size,
        sample=1e-4,
        negative=10,
        min_count=min_count,
    )
    model.save(vector_out_base + ".pickle")
    model.wv.save_word2vec_format(
        vector_out_base + ".vectors", fvocab=vector_out_base + ".voc"
    )


def main(input_pattern, event_emb_out_base, min_count=5):
    print("Input pattern is " + input_pattern)
    print("Output base is " + event_emb_out_base)

    if not os.path.exists(event_emb_out_base + ".vectors"):
        print("Training embeddings.")

        embedding_dir = os.path.dirname(event_emb_out_base)
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)

        train_event_vectors(
            input_pattern, event_emb_out_base, window_size=10, min_count=min_count
        )


if __name__ == "__main__":
    import sys

    in_pattern = sys.argv[1]
    output_base = sys.argv[2]

    if len(sys.argv) == 4:
        min_count = int(sys.argv[3])
    else:
        min_count = 5

    main(in_pattern, output_base, min_count)
