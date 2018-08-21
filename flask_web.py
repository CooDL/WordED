from flask import Flask, request, render_template, send_from_directory
import time
from backend import WEDemo, plot_figure, plot_3D_figure
import numpy as np
# import sys
# sys.setrecursionlimit(10000)

word2vecs = WEDemo.load_word2vec_format('./glove.6B.50d.w2v.txt') # load pre-trained embeddings

app = Flask(__name__)

UPLOAD_FOLDER = './images' # images path

app.config.update( # update some config of flask
    dict(DEBUG=True,
    SECRET_KEY='development key',
    TIME_STAMP=int(time.time()),
    UPLOAD_FOLDER=UPLOAD_FOLDER
))

# ===================================Define Routes and Actions==================================
@app.route('/')
def getindex():
    '''
    the render function for the '/' route
    :return: the function to render a template file
    '''
    return render_template('index.html')


@app.route('/wed')
def getweb_r():
    '''
    the render function for the '/wed' route
    :return: the function to render a template file
    '''
    return render_template('wed.html')


@app.route('/images/<filename>')
def send_file(filename):
    '''
    the function to send the image files
    :param filename: the filename to be sent
    :return: the function to send the picture file
    '''
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/wed', methods=['POST']) # the function to get and answer the query from web
def getweb():
    '''
    'the function to get and answer the query from web'
    :return: the render values for the html templates
    '''
    keyword = request.form['keyword']
    top_k = request.form['top_k']
    near, vocabs, probs= word2vecs.similar_by_word_r(keyword, topn=int(top_k))
    fig_name_2D = 'fig_single_2D_'+keyword+'_'+str(top_k)+'.png'
    fig_name_3D = 'fig_single_3D_'+keyword+'_'+str(top_k)+'.png'
    vocabs = vocabs + [keyword]
    probs = probs + [1.0]
    embeddings = np.array([word2vecs[itm] for itm in vocabs])

    # plot figures
    plot_figure(vocabs=vocabs, embeddings=embeddings, dims=2, fig_name='./images/'+fig_name_2D)
    plot_3D_figure(vocabs=vocabs, embeddings=embeddings, probs=probs, fig_name='./images/'+fig_name_3D)
    print(near)
    return render_template('weda.html', entries=near, keyword=keyword, top_k=top_k, figname1=fig_name_2D, figname2=fig_name_3D)


@app.route('/weds')
def getwebs_r():
    '''
    # the render function for the '/weds' route
    :return: the render values for the html templates
    '''
    return render_template('weds.html')

@app.route('/weds', methods=['POST'])
def getwebs():
    '''
    'the function to get and answer the query from web'
    :return: the render values for the html templates
    '''
    keyword1 = request.form['keyword1']
    keyword2 = request.form['keyword2']
    step_k = request.form['step_k']
    top_k = request.form['top_k']
    results, bvocabs, probs = word2vecs.find_between(keyword1, keyword2, int(step_k))
    nears, svocabs, probss = word2vecs.similar_by_word_rs(keyword1, keyword2, topn=int(top_k))
    fig_name_2D = 'fig_double_2D_'+keyword1+'_'+keyword2+'_'+str(step_k)+'.png'
    fig_name_2Ds = 'fig_double_2D_' + keyword1 + '_' + keyword2 + '_' + str(top_k) + 's.png'
    fig_name_3D = 'fig_double_3D_'+keyword1+'_'+keyword2+'_'+str(step_k)+'.png'
    fig_name_3Ds = 'fig_double_3D_' + keyword1 + '_' + keyword2 + '_' + str(top_k) + 's.png'
    vocabs = bvocabs + [keyword1, keyword2]
    vocabss = svocabs + [keyword1, keyword2]
    probs = probs + [1.0, 1.0]
    probss = probss + [1.0, 1.0]
    print(bvocabs, results)
    embeddings = np.array([word2vecs[itm] for itm in vocabs])
    embeddingss = np.array([word2vecs[itm] for itm in vocabss])
    plot_figure(vocabs=vocabs, embeddings=embeddings, dims=2, fig_name='./images/'+fig_name_2D)
    plot_figure(vocabs=vocabss, embeddings=embeddingss, dims=2, fig_name='./images/' + fig_name_2Ds)
    plot_3D_figure(vocabs=vocabs, embeddings=embeddings, probs=probs, fig_name='./images/'+fig_name_3D)
    plot_3D_figure(vocabs=vocabss, embeddings=embeddingss, probs=probss, fig_name='./images/' + fig_name_3Ds)
    return render_template('wedsa.html', keyword1=keyword1, keyword2=keyword2, top_k=top_k, step_k=step_k,
                             entries=nears, betweens=results, figsname=fig_name_2Ds, fig3sname=fig_name_3Ds, figname1=fig_name_2D, figname2=fig_name_3D)#betweens=results


if __name__ == '__main__':
    app.run(port=5200)
