import tensorflow as tf
import logging

logger = logging.getLogger()


if __name__ == '__main__':
    x = tf.Variable(3, name='x')
    y = tf.Variable(4, name='y')
    f = x * x * y + y + 2

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        result = f.eval()
        print(result)
