import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 

tf.set_random_seed(1)
np.random.seed(1)

# fake data
n_data = np.ones((300, 2))
# print(n_data)
x0 = np.random.normal(2*n_data, 1)
y0 = np.zeros(300)

x1 = np.random.normal(-2*n_data, 1)
y1 = np.ones(300)

x = np.vstack((x0, x1))
y = np.hstack((y0, y1))

# plot data
plt.scatter(x[:, 0], x[:, 1], c=y, lw=0, marker='s', edgecolors='white')
plt.show()

# placeholder
tf_x = tf.placeholder(tf.float32, shape=x.shape)
tf_y = tf.placeholder(tf.int32, shape=y.shape)

# neural network
hidden = tf.layers.dense(tf_x, 10, tf.nn.relu)
output = tf.layers.dense(hidden, 2)

loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)
accuracy = tf.metrics.accuracy(labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1))[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
	sess.run(init_op)
	plt.ion()
	for step in range(200):
		# train and net output
		_, acc, pred = sess.run([train_op, accuracy, output], feed_dict={tf_x:x, tf_y:y})
		if step % 2 == 0:
			plt.cla()
			plt.scatter(x[:, 0], x[:, 1], c=pred.argmax(1), marker='s', lw=0, edgecolors='white')
			plt.text(1.5, -4, 'accuracy:%.2f' %acc, fontdict={'size':20, 'color':'blue'})
			plt.pause(0.1)

	plt.ioff()
	plt.show()
