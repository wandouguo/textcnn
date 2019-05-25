import os, sys
import tensorflow as tf

base_dir = os.path.dirname(__file__)
sys.path.insert(0, base_dir)
from data.dataset import *
from cnn_model import *

batch_size = 64
train_file = "../data/cnews.train.txt"
vocab_file = "../data/cnews.vocab.txt"
words, word2id = read_vocab(vocab_file)
vocab_size = 5000
category, cat2id = read_category()
max_seq_len = 600
x_pad, y_pad = process_file(train_file, word2id, cat2id, max_seq_len)
model = CnnModel(None, vocab_size, batch=batch_size, sequence_max_len=max_seq_len)
session = model.session
opt, loss, pred, acc = model.build_model()
init = tf.global_variables_initializer()
session.run(tf.local_variables_initializer())
session.run(tf.initialize_all_variables())
session.run(init)

for epoch in range(4):

    print("epoch ", epoch)
    i = 0
    for input_batch, label_batch in batch_iter(x_pad, y_pad, batch_size):
        # print(len(input_batch, label_batch)
        _, batch_loss, batch_pred, batch_acc = session.run([opt, loss, pred, acc],
                                                           feed_dict={model.input: input_batch,
                                                                      model.label: label_batch})
        print("step={},loss={},epoch={},acc={}".format(i, batch_loss, epoch, batch_acc))
        i = i + 1
