import tensorflow as tf

if __name__ == '__main__':
    w = tf.constant(3)
    x = w + 2
    y = x + 5
    z = x * 3

    # By default TF does not reuse node values, x and w will be evaluated
    # twice
    with tf.Session() as sess:
        print(y.eval())  # 10
        print(z.eval())  # 15

    # To evaluate z and y efficiently run
    with tf.Session() as sess:
        y_val, z_val = sess.run([y, z])
        print(y_val)
        print(z_val)
