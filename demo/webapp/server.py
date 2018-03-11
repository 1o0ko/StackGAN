import numpy as np

import cStringIO as StringIO
from skimage.io import imsave

from flask import Flask, render_template, request
from flask import send_file


from misc.config import cfg, cfg_from_file
from embedding.model import Model
from embedding.preprocessing import normalize
from demo.demo_custom_embeddings import parse_args, GenerativeModel 

app = Flask(__name__)

text_model = None
img_model = None
NUM_IMGS = 8

def embed_text(text, text_model):
    texts = [normalize(str(text))]
    embeddings, num_embeddings = text_model.embed(texts), len(texts)
    print('Total number of sentences:', num_embeddings)
    print('num_embeddings:', num_embeddings, embeddings.shape)

    return embeddings, num_embeddings, texts


@app.route('/')
def my_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text'].lower()
    embeddings, num_embeddings, captions_list = embed_text(text, text_model)
    imgs = img_model.generate(embeddings, captions_list, n=NUM_IMGS)
    print('Generated: %d images' % len(imgs))
    strIO = StringIO.StringIO()
    imsave(strIO, imgs[0], plugin='pil', format_str='png')
    strIO.seek(0)

    return send_file(strIO, mimetype='image/jpeg')


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
        print(cfg)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id

    text_model = Model(
        '/models/fashion/frozen_model.pb',
        '/models/fashion/tokenizer.pickle')

    img_model = GenerativeModel(cfg, 1, 1024)

    app.run(host='0.0.0.0', port=8080)
