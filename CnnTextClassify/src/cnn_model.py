import numpy as np
import tensorflow as tf


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
            self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.word_dim],
                                             initializer=tf.random_normal_initializer(mean=0, stddev=0.02))

        self.cnn_kernel = [5, self.word_dim, 1, 3]  # 目的在于保证在word_dim方向不移动
        self.pool_kernel = [1, 5, 1, 1]
        self.input = tf.placeholder(shape=(self.batch, self.max_len), dtype=tf.int32)
        self.label_num = 10
        self.label = tf.placeholder(shape=(self.batch, self.label_num))
        pass

    def embedding_layer(self, input, name="embedding"):
        with tf.name_scope(name=name):
            return tf.nn.embedding_lookup(self.embedding, input, name=name)

    def dropout_layer(self, input, drop_rate=0.1, name="dropout"):
        with tf.name_scope(name=name):
            return tf.nn.dropout(input, 1 - drop_rate, name=name)

    def cnn_layer(self, input, kernel, name="cnn"):
        ## input =[batch,tex_lenth,embbeding_dim]
        ## kernel =[heitht,with,kernerl_number]
        # return : [batch,out_heigth,out_with,kernel_number]
        with tf.name_scope(name=name):
            cnn_input = tf.expand_dims(input, axis=-1)
            return tf.nn.conv2d(cnn_input, kernel, [1, 1, 1, 1], name=name)

    def maxpool_layer(self, input, pool_kernel, name="max_pool"):
        with tf.name_scope(name=name):
            return tf.nn.max_pool(input, pool_kernel, [1, 1, 1, 1], padding="VALID", name=name)

    def dense_lyaer(self, input, hidden_dim, name="dense"):
        with tf.name_scope(name=name):
            tf.layers.dense(input, hidden_dim, name=name)
        pass

    def build_model(self):

        embedding = self.embedding_layer(self.input)
        embedding_dropout = self.dropout_layer(embedding)
        cnn_out = self.cnn_layer(embedding_dropout, self.cnn_kernel)
        max_out = self.maxpool_layer(cnn_out, self.pool_kernel)  # [batch, max_heigth,1,max_number]
        max_out = tf.squeeze(max_out, axis=-2)  ## [batch, max_heigth,max_number]
        self.logits = self.dense_lyaer(max_out.reshape(self.batch, None), self.label_num)
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
        loss = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.label)
        self.loss = tf.reduce_sum(loss)
        opt = tf.train.AdamOptimizer(loss)
        return opt, self.loss, self.y_pred_cls


if __name__ == "main":
    model = CnnModel(None, vocab_size=5)
    model.build_model()
