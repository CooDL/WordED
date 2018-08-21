from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib, os
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mplot3d import Axes3D
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


class WEDemo(KeyedVectors):
    ''''''
    # ==============Find near top-n words by a given word==========
    def similar_by_word_r(self, word, topn=10, restrict_vocab=None):
        '''
        :param word: keyword
        :param topn: the topn result similar to keyword
        :param restrict_vocab: limits the range; For example, restrict_vocab=10000 would
        only check the first 10000 word vectors in the vocabulary order.
        :return: a zip of topn and corresponing result
        '''
        try:
            results = self.similar_by_word(word=word, topn=topn, restrict_vocab=restrict_vocab)
            vocabs = [res for res, _ in results]
            probs = [rse for _, rse in results]
            return [zip(range(topn), list(results)), vocabs, probs]
        except KeyError as e:
            print('Unknown word %s'%word)
            return [zip([0], [(word, 1)]), [word], [1]]

    # ==============Find near top-n words by a given word==========
    def similar_by_word_rs(self, word1, word2, topn=10, restrict_vocab=None):
        '''
        :param word1: keyword1
        :param word2: keyword1
        :param topn: the topn result similar to keyword1 & keyword2
        :param restrict_vocab: limits the range;
        :return: a zip of topn and corresponing results
        '''
        try:
            results1 = self.similar_by_word(word=word1, topn=topn, restrict_vocab=restrict_vocab)
            results2 = self.similar_by_word(word=word2, topn=topn, restrict_vocab=restrict_vocab)
            vocabs = [res for res, _ in (results1+results2)]
            probs = [rse for _, rse in (results1+results2)]
            return zip(range(topn), list(results1), list(results2)), vocabs, probs
        except KeyError as e:
            print('Unknown word')
            return [zip([0], [(word1, 1)], [(word2, 1)]), [word1, word2], [1]]

    # ==============Find near top-n words by a given vector==========
    def similar_by_vector_r(self, vector, topn=1, restrict_vocab=None):
        '''
        :param vector: A vector has the same dimension with word_embedding
        :param topn: the topn result similar to vector
        :param restrict_vocab: limits the range;
        :return: the topn words near the vector
        '''
        results = self.similar_by_vector(vector=vector, topn=topn, restrict_vocab=restrict_vocab)
        return results

    # ==============Find between words by the given keywords and the step value==========
    def find_between(self, word1, word2, step):
        '''
        :param word1: keyword1
        :param word2: keyword2
        :param step: how many step between word1 and word2. For example, if step is 1, the vector
        would be the word similar to the average vector the two vector, v1 = vector(word1) +
        1/2(vector(word2)-vector(word2))
        :return: the word most similar to the step vector
        '''
        try:
            vector1 = self[word1]
            vector2 = self[word2]
        except KeyError:
            print('Unknown Words')
            return zip([0], [('word1', 1)]), [word1, word2], [1]
        distance = vector2 - vector1
        vocabs = []
        word_list = []
        probs = []
        for idx in range(1, step+1):
            vect = vector1 + idx*distance/float(step+1)
            results = self.most_similar(positive=[vect], topn=2)
            print(results)
            word_list.append([idx/float(step+1), list(results[1])])
            vocabs.append(results[1][0])
            probs.append(results[1][1])
        return zip(range(step+1), word_list), vocabs, probs

    def format_vocab_embedding(self, vocabs):
        '''
        :param vocabs: the words list
        :return: the zip format of the words list and its looking-up results
        '''
        vectors = []
        for vocab in vocabs:
            try:
                vectors.append(self[vocab])
            except KeyError:
                print('Unknown word %s'%vocab)
                vectors.append(np.zeros(self.vectors.shape[-1]))
        return vocabs, np.array(vectors)

# ===============Functions for Plotting Figures============================================
# ==============Plot 2D Figure==========
def plot_figure(vocabs, embeddings, dims=2, fig_name='tsne.png'):
    '''
    :param vocabs: the vocabs to print
    :param embeddings: the vectors of the words to be printed
    :param dims: the dimension of the figure
    :param fig_name: tmp filename
    :return:
    '''
    def plot_with_labels(low_dim_embs, labels, filename):
        '''
        :param low_dim_embs: the vectors of labels
        :param labels: the words to be printed
        :param filename: the tmp filename
        :return:
        '''
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(8, 8))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
        plt.savefig(filename)
        plt.close()
        print("Finished")
    try:
        # pylint: disable=g-import-not-at-top
        tsne = TSNE(perplexity=30, n_components=dims, init='pca', n_iter=5000, method='exact')
        plot_only = len(vocabs)
        low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
        labels = [vocabs[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels, fig_name)
    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)
# ==============Plot 3D Figure==========
def plot_3D_figure(vocabs, embeddings, probs, fig_name='tsne.png'):
    '''
    :param vocabs: the vocabs to print
    :param embeddings: the vectors of the words to be printed
    :param probs: the similarities of the words to the quried word
    :param fig_name: tmp filename
    :return:
    '''
    tsne = TSNE(perplexity=30, n_components=3, init='pca', n_iter=5000, method='exact')
    plot_only = len(vocabs)
    fig = plt.figure(figsize=(20, 12))
    ax = fig.gca(projection='3d')
    #ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    transformed_vec = tsne.fit_transform(embeddings[:plot_only, :]) # squash to 3D
    xdata, ydata, zdata = transformed_vec[:, 0], transformed_vec[:, 1], transformed_vec[:, 2]
    for i in range(len(vocabs)):
        ax.text(xdata[i], ydata[i], zdata[i], '%s'%vocabs[i], size=16, zorder=1, color='k') # add label
    ax.scatter3D(xdata, ydata, zdata, c='r', marker='^', s=np.sqrt(np.square(zdata))*np.array(probs))
    # ax.plot(xdata, ydata, zdata, color='r' )
    plt.savefig(fig_name)
    plt.close()
    print("Finished")

def _start_shell(local_ns=None):
    '''
    :param local_ns: a function to start the shell to debug backend.py
    :return:
    '''
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)

if __name__ == '__main__':
    word2vecs = WEDemo.load_word2vec_format('./glove.6B.50d.w2v.txt')
    _start_shell()
