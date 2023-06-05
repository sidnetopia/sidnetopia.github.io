import pickle

import numpy as np
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from token_embedding import TokenEmbedding

def get_largest_indices(lst, k):
    k = -1 * k
    top_indices = np.argpartition(lst, k)[k:]

    return top_indices

def cos_sim(W, x, k):
    A = W
    B = x.reshape(-1,)
    A_norm = np.linalg.norm(A, 2)
    B_norm = np.linalg.norm(B, 2)
    print(A)
    print(B)
    print(A_norm)
    print(B_norm)
    cos = np.dot(A, B) / (A_norm * B_norm) + 1e-9 # avoid division by 0

    topk = get_largest_indices(cos, k)

    return topk, [cos[int(i)] for i in topk]

def get_similar_tokens(query_token, k, embed):
    if np.all(embed[[query_token]] == 0):
        raise "Division by zero"

    topk, cos = cos_sim(embed.idx_to_vec, embed[[query_token]], k + 1)

    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'{float(c):.3f}= {embed.idx_to_token[int(i)]}')

def get_analogy(token_a, token_b, token_c, k, embed):
    # token_a - token_b = token_c - token_d(?)
    # token_b - token_a + token_c = token_d(?)

    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = cos_sim(embed.idx_to_vec, x, k)

    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'{float(c):.3f}: {embed.idx_to_token[int(i)]}')

def print_from_scratch(embedding, algorithm_title):
    print(f"From scratch word cosine similarity trained on {algorithm_title}")
    print("Word similarity:")
    for word in word_similarity_list:
        try:
            print(f"\nWord:{word}")
            get_similar_tokens(word, 10, embedding)
        except:
            print(f"Word {word} not found in embedding")
    print("")

    print("Word analogy:")
    for word in word_analogy_list:
        try:
            print(f"\n{word[0]} is to {word[1]}, as {word[2]} is to __: ")
            get_analogy(word[0], word[1], word[2], 10, embedding)
            print(f"answer: {word[3]}")
        except KeyError as e:
            print(f"Words {e.args} not found in embedding")

def print_from_gensim(embedding, algorithm_title):
    print(f"Using gensim library, word cosine similarity trained on {algorithm_title}")
    print("Word similarity:")
    word_similarities = {}
    for word in word_similarity_list:
        try:
            print(f"\nWord:{word}")
            sim = embedding.most_similar(positive=[word])
            print(sim)
            word_similarities[word] = sim
        except KeyError as e:
            print(f"Word {e.args[0]} not found in embedding")
    print("")

    print("Word analogy:")
    word_analogies = {}
    for word in word_analogy_list:
        try:
            print(f"\n{word[0]} is to {word[1]}, as {word[2]} is to __: ")
            word_analogy = embedding.most_similar(negative=[word[0]], positive=[word[1], word[2]])
            print(word_analogy)
            print(f"answer: {word[3]}")
            word_analogies[word[3]] = word_analogy
        except KeyError as e:
            print(f"Words {e.args} not found in embedding")
    file = open(f"word_sim_analogy.obj_{algorithm_title}", 'wb')
    pickle.dump({
        "word_similarities": word_similarities,
        "word_analogies":word_analogies
                 },
        file)

    return word_similarities, word_analogies

def plot_data(orig_data, labels):
    pca = PCA(n_components=2)
    data = pca.fit_transform(orig_data)
    plt.figure(figsize=(7, 5), dpi=100)
    plt.plot(data[:, 0], data[:, 1], '.')
    for i in range(len(data)):
        plt.annotate(labels[i], xy=data[i])
    for i in range(len(data) // 2):
        plt.annotate("",
                     xy=data[i],
                     xytext=data[i + len(data) // 2],
                     arrowprops=dict(arrowstyle="->",
                                     connectionstyle="arc3")
                     )



if __name__ == '__main__':
    embedding_dir = '/home/sidnetopia/Documents/nlp/assignments/word_embedding/embeddings/oscar_tl'
    word_similarity_list = [
        "baba",
        "basa",
        "babae",
        "ako",
        "ospital",
        "hospital",
        "Marcos",
        "Piolo",
        "umaga",
        "kape",
    ]

    word_analogy_list = [
        ('umaga', 'breakfast', 'gabi', ''),  # near is to far, as open is to close
        ('lalaki', 'tatay', 'babae', ''),  # school is to teacher, as hospital is to doctor
        ('ospital', 'sakit', 'bahay', ''),  # hammer is to nail, as comb is to hair
        ('kape', 'mainit', 'kain', ''),  # hand write mouth read
        ('ulan', 'bagyo', 'araw', ''),  # dog is to puppy, as cat is to kitten
    ]

    # oscar_fasttext = TokenEmbedding(f'{embedding_dir}/oscar_w2v_fasttext')
    # oscar_word2vec = TokenEmbedding(f'{embedding_dir}/oscar_w2v_word2vec')
    # print_from_scratch(oscar_fasttext, 'fastText')
    # print("")
    # print_from_scratch(oscar_word2vec, 'word2vec')
    # print("")

    oscar_fasttext_gensim = KeyedVectors.load(f'{embedding_dir}/oscar_tl_fasttext.vec')
    oscar_word2vec_gensim = KeyedVectors.load(f'{embedding_dir}/oscar_tl_word2vec.vec')
    print_from_gensim(oscar_fasttext_gensim, 'fastText')
    print("")
    print_from_gensim(oscar_word2vec_gensim, 'word2vec')

