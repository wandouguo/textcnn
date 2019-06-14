import numpy as np
import tensorflow as tf
from data.dataset import *


class CnnModel(object):
    def __init__(self, embedding, vocab_size, batch=512, sequence_max_len=32):
        self.batch = batch
        self.max_len = sequence_max_len
        if embedding:
            self.embedding = embedding
            self.word_dim = self.embedding.shape[1]
            self.vocab_size = self.embedding.shape[0]
        else:
            self.vocab_size = vocab_size
            self.word_dim = 128
            self.embedding = tf.get_variable("embedding", shape=(self.vocab_size, self.word_dim),
                                             dtype=tf.float32,
                                             trainable=True)

        self.cnn_kernel = [5, self.word_dim, 1, 256]  # 目的在于保证在word_dim方向不移动
        self.pool_kernel = [1, self.max_len - 5 + 1, 1, 1]
        self.input = tf.placeholder(shape=(self.batch, self.max_len), dtype=tf.int32)
        self.label_num = 10
        self.hidden_num = 128
        self.label = tf.placeholder(shape=(self.batch, self.label_num), dtype=tf.int32)
        self.dropout_rate = tf.placeholder(dtype=tf.float32)
        self.session = tf.Session()
        pass

    def embedding_layer(self, input, name="embedding"):
        with tf.name_scope(name=name):
            return tf.nn.embedding_lookup(self.embedding, input, name=name)

    def dropout_layer(self, input, drop_rate, name="dropout"):
        with tf.name_scope(name=name):
            return tf.nn.dropout(input, 1 - drop_rate, name=name)

    def cnn_layer(self, input, kernel, name="cnn"):
        ## input =[batch,tex_lenth,embbeding_dim]
        ## kernel =[heitht,with,kernerl_number]
        # return : [batch,out_heigth,out_with,kernel_number]
        with tf.name_scope(name=name):
            # cnn_input = tf.expand_dims(input, axis=-1)
            # `[batch, in_height, in_width, in_channels]
            return tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding="VALID", name=name)

    def maxpool_layer(self, input, pool_kernel, name="max_pool"):
        with tf.name_scope(name=name):
            return tf.nn.max_pool(input, pool_kernel, [1, 1, 1, 1], padding="VALID", name=name)

    def dense_lyaer(self, input, hidden_dim, name="dense"):
        with tf.name_scope(name=name):
            return tf.layers.dense(input, hidden_dim, name=name)

    def build_model(self):

        embedding = self.embedding_layer(self.input)
        print("embedding ", embedding.shape)
        cnn_input = tf.expand_dims(embedding, axis=-1)
        print("cnn_input ", cnn_input.shape)
        # embedding_dropout = self.dropout_layer(embedding, self.dropout_rate)
        kernel = tf.get_variable("kernel_filter", shape=self.cnn_kernel,
                                 dtype=tf.float32)
        cnn_out = self.cnn_layer(cnn_input, kernel, name="cnn")
        print("cnn_out ", cnn_out.shape)
        max_out = self.maxpool_layer(cnn_out, self.pool_kernel)  # [batch, max_heigth,1,max_number]
        print("max_out ", max_out.shape)
        max_out = tf.squeeze(max_out, axis=-2)  ## [batch, max_heigth,max_number]
        print("max_out ", max_out.shape)
        input = tf.squeeze(max_out, axis=-2)
        print("input ", input.shape)
        dense = self.dense_lyaer(input, self.hidden_num, name="fc")
        drop_out = self.dropout_layer(dense, self.dropout_rate, name="dense_drop")
        fc = tf.nn.relu(drop_out)
        print("fc ", fc.shape)
        self.logits = self.dense_lyaer(fc, self.label_num)
        soft_max = tf.nn.softmax(self.logits)
        y_pred_cls = tf.argmax(soft_max, 1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
        correct_pred = tf.equal(tf.argmax(self.label, 1), y_pred_cls)
        loss = tf.reduce_mean(loss)

        opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss)

        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return opt, loss, y_pred_cls, acc


if __name__ == '__main__':
    model = CnnModel(None, vocab_size=5)
    model.build_model()
    x = [[[1, 2, 3],
          [4, 3, 1]],
         [[3, 2, 1],
          [2, 3, 1]]]

    x = np.array(x)
    print(x, x.shape)
    y = np.reshape(x, (2, -1))
    print(y, y.shape)
    # tf.nn.conv1d()
    # tf.layers.conv1d()
    z = (3,) * 3
    print("z", z)
    kernel_shape = (3,) + (2, 3)
    print(kernel_shape)
