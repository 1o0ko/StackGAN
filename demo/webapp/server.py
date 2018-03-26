import os
import cStringIO as StringIO
from skimage.io import imsave

from flask import Flask, render_template, request
from flask import send_file

from demo.demo_embeddings_tf import parse_args, save_super_images, embed_text
from demo.demo_embeddings_tf import GenerativeModel
from embedding.model import Model
from misc.config import cfg, cfg_from_file

app = Flask(__name__)

text_model = None
img_model = None
NUM_IMGS = 8
MODEL_PATH = '/models/fashion/embedding_model'


@app.route('/')
def my_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text'].lower()
    embeddings, num_embeddings, normalized_texts = embed_text([str(text)], text_model)
    hr_imgs, lr_imgs = img_model.generate_n(embeddings, n=NUM_IMGS)
    imgs = save_super_images(lr_imgs, hr_imgs, normalized_texts, 1, startID=0)

    print('Generated: %d images' % len(hr_imgs))
    strIO = StringIO.StringIO()
    imsave(strIO, imgs[0], plugin='pil', format_str='png')
    strIO.seek(0)

    return send_file(strIO, mimetype='image/jpeg')


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.caption_path is None:
        args.caption_path = MODEL_PATH

    text_model = Model(
        os.path.join(args.caption_path, 'frozen_model.pb'),
        os.path.join(args.caption_path, 'tokenizer.pickle'))

    img_model = GenerativeModel(cfg, 1, 1024)

    app.run(host='0.0.0.0', port=8080)
