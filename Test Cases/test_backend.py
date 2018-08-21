import sys, os, argparse
import numpy as np
sys.path.insert(0, '../')
from backend import plot_3D_figure, plot_figure, WEDemo, _start_shell

params = argparse.ArgumentParser()
params.add_argument('--rad', '-r', type=bool, default=False)
params.add_argument('--topk', '-t', type=int, default=5, help='the top-k words')
params.add_argument('--stepk', '-s', type=int, default=5, help='the step number')
args = params.parse_args()

if not os.path.exists('./../glove.6B.50d.w2v.txt'):
    os.system('./../getGloVe.sh')
word2vecs = WEDemo.load_word2vec_format('./../glove.6B.50d.w2v.txt')
if args.rad is False:
    samples1 = ['today', 'is', 'a', 'sunny', 'day', 'happy']
    samples2 = ['wdwadwa', 'asdsschool', 'figawfas', 'adoapwdnafu', 'waduawdf', 'wadaufea']
    samples3 = ['vocation', 'benefit', 'our', 'healthy', 'quite', 'interesting']
    print('Start testing the backend')
    print('\n************************************\n1a.Testing function with normal words: \tsimilar_by_word_r\nTest samples %s'%str(samples1))

    for word in samples1:
        print('\n#####SAMPLE########\n%s'%word)
        print(word2vecs.similar_by_word_r(word, topn=args.topk))

    print('\n************************************\n1b.Testing function with unknown words: \tsimilar_by_word_r\nTest samples %s'%str(samples2))
    for word in samples2:
        print('\n#####Unkown SAMPLE########\n%s'%word)
        print(word2vecs.similar_by_word_r(word, topn=args.topk))

    print('\n************************************\n2a.Testing function with normal words: \tsimilar_by_word_rs\nTest samples %s'%str(samples2))
    for word1, word2 in zip(samples1, samples3):
        print('\n#####SAMPLE########\n%s and %s'%(word1, word2))
        print(word2vecs.similar_by_word_rs(word1, word2, topn=args.topk))

    print('\n************************************\n2b.Testing function with unknown words: \tsimilar_by_word_rs\nTest samples %s'%str(samples2))
    for word1, word2 in zip(samples2, samples2):
        print('\n#####Unknown SAMPLEs########\n%s and %s'%(word1, word2))
        print(word2vecs.similar_by_word_rs(word1, word2, topn=args.topk))

    print('\n************************************\n3.Testing function: \tsimilar_by_vector_r')
    sampleE = []
    for iword in samples1:
        try:
            sampleE.append(word2vecs[iword])
        except KeyError:
            print('')
    print('Test samples \n', np.array(sampleE))
    for warry in sampleE:
        print('\n#####SAMPLE########\n', warry)
        print(word2vecs.similar_by_vector_r(warry, topn=args.topk))

    print('\n************************************\n4a.Testing function with normal words: \tfind_between\nTest samples %s and %s'%(str(samples3), str(samples1)))
    for word1, word2 in zip(samples3, samples1):
        print('\n#####SAMPLEs########\n%s and %s'%(word1, word2))
        print(word2vecs.find_between(word1, word2, step=args.stepk))

    print('\n************************************\n4b.Testing function with unknown words: \tfind_between\nTest samples %s and %s'%(str(samples2), str(samples1)))
    for word1, word2 in zip(samples2, samples1):
        print('\n#####Unknown SAMPLEs########\n%s and %s'%(word1, word2))
        print(word2vecs.find_between(word1, word2, step=args.stepk))

    print('\n************************************\n5a.Testing function with normal words: \tformat_vocab_embed ding\nTest samples %s'%str(samples1))
    print('\n#####SAMPLEs########\n%s'%str(samples1))
    print(word2vecs.format_vocab_embedding(samples1))

    print('\n************************************\n5b.Testing function with unknown words: \tformat_vocab_embed ding\nTest samples %s'%str(samples2))
    print('\n#####SAMPLEs########\n%s'%str(samples2))
    print(word2vecs.format_vocab_embedding(samples2))

    print('\n************************************\n6.Testing function: \tplot_figure\nTest samples %s'%str(samples1))
    vocabs, embeddings = word2vecs.format_vocab_embedding(samples1)
    print(plot_figure(vocabs, embeddings))

    print('\n************************************\n7.Testing function: \tplot_3D_figure\nTest samples %s' % str(samples1))
    print(plot_3D_figure(vocabs, embeddings, probs=range(len(samples1)), fig_name='tsne.3d.png'))

else:
    samples = list(set(open('sample.txt', 'r').read().replace('\n', ' ').split()))
    print('Start testing the backend')
    print('\n************************************\n1ab.Testing function: \tsimilar_by_word_r\nTest samples %s'%str(samples))

    for word in samples:
        print('\n#####SAMPLE########\n%s'%word)
        print(word2vecs.similar_by_word_r(word, topn=args.topk))

    print('\n************************************\n2ab.Testing function: \tsimilar_by_word_rs\nTest samples %s'%str(samples))
    for word1, word2 in zip(samples, samples):
        print('\n#####SAMPLE########\n%s and %s'%(word1, word2))
        print(word2vecs.similar_by_word_rs(word1, word2, topn=args.topk))

    print('\n************************************\n3.Testing function: \tsimilar_by_vector_r')
    sampleE = []
    for iword in samples:
        try:
            sampleE.append(word2vecs[iword])
        except KeyError:
            print('')
    print('Test samples \n', np.array(sampleE))
    for warry in sampleE:
        print('\n#####SAMPLE########\n', warry)
        print(word2vecs.similar_by_vector_r(warry, topn=args.topk))

    print('\n************************************\n4ab.Testing function: \tfind_between\nTest samples %s and %s'%(str(samples), str(reversed(samples))))
    for word1, word2 in zip(samples, reversed(samples)):
        print('\n#####SAMPLEs########\n%s and %s'%(word1, word2))
        print(word2vecs.find_between(word1, word2, step=args.stepk))

    print('\n************************************\n5ab.Testing function with normal words: \tformat_vocab_embed ding\nTest samples %s'%str(samples))
    print('\n#####SAMPLEs########\n%s'%str(samples))
    print(word2vecs.format_vocab_embedding(samples))

    print('\n************************************\n6.Testing function: \tplot_figure\nTest samples %s'%str(samples))
    vocabs, embeddings = word2vecs.format_vocab_embedding(samples)
    print(plot_figure(vocabs, embeddings))

    print('\n************************************\n7.Testing function: \tplot_3D_figure\nTest samples %s' % str(samples))
    print(plot_3D_figure(vocabs, embeddings, probs=range(len(samples)), fig_name='tsne.3d.png'))


_start_shell()
