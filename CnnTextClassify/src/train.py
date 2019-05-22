import os, sys
import tensorflow as tf

base_dir = os.path.dirname(__file__)
sys.path.insert(0, base_dir)
from cnn_model import CnnModel

if __name__ == "main":
    model = CnnModel()
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    opt, loss, pred = model.build_model()
    print('hello')
