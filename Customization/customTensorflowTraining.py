#
# 自定义的训练
#

import tensorflow as tf

# f(x) = ax + b

a = tf.Variable(tf.constant(5, dtype=tf.float32))

lr = 0.2

epoch = 50

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        loss = tf.square(a + 1)
    grads = tape.gradient(loss, a)

    a.assign_sub(lr * grads)

    print("After %s epoch,a is %f,loss is %f" % (epoch, a.numpy(), loss))
