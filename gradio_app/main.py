import gradio as gr
import numpy  as np
import tensorflow as tf

from PIL import Image

model = tf.keras.models.load_model('model')

def predict_model(im):
    im = tf.image.convert_image_dtype(im,tf.float32)
    im = tf.expand_dims(im,axis=0)
    
    pred = model.predict(im)
    pred = pred[0]
    pred = tf.clip_by_value(pred, 0.0, 1.0)
    
    pred = Image.fromarray((np.array(pred) * 255).astype(np.uint8))
    return pred

inimage = gr.Image(shape=(None,None))
inimage.style(width=512,height=512)

outimage = gr.Image(shape=(None,None))
outimage.style(width=512,height=512)

app = gr.Interface(fn=predict_model, inputs=inimage, outputs=outimage,examples=['./assets/examples/creature.jpg',
                                                                                "./assets/examples/butterfly.jpeg",
                                                                                "./assets/examples/bw.jpg",
                                                                                "./assets/examples/hillbefore.jpg"])

app.launch()

