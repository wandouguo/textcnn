import numpy as np
import tensorflow as tf

embedding = np.eye(6)
input = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 1], [0, 1, 2, 3, 4], [2, 3, 4, 5, 0]]

session = tf.Session()

em_out = tf.nn.embedding_lookup(embedding, input)
# expand_em = tf.expand_dims(em_out, axis=-1)
xx = session.run(em_out)
# print(xx.shape)
cnn_input = tf.expand_dims(em_out, axis=-1)
print(tf.shape(cnn_input))
kernel = np.ones(shape=(2, 6, 2))
cnn_kernel = tf.expand_dims(kernel, axis=-2)
yy = tf.nn.conv2d(cnn_input, cnn_kernel, [1, 1, 1, 1], padding="VALID")
con_out = session.run(yy)
# print(con_out, con_out.shape)
zz = tf.nn.max_pool(yy, [1, 2, 1, 1], [1, 1, 1, 1], padding="VALID")
zz_ = session.run(zz)
print(zz_, zz_.shape)
concat_out = tf.squeeze(zz, axis=-2)
cat_out = session.run(concat_out)
print("cat:", cat_out, cat_out.shape)
denc = tf.layers.dense(concat_out, 1)
session.run(tf.global_variables_initializer())
cat2 = tf.reduce_sum(denc, axis=-1)
print(session.run(cat2).shape, session.run(cat2))
dense = session.run(denc)
print(dense, dense.shape)
