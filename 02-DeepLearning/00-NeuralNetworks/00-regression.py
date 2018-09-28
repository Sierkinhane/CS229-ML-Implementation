import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 

# fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis]  # shape=(200, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise

# plt.scatter(x, y)
# plt.show()

# placeholder
tf_x = tf.placeholder(tf.float32, shape=x.shape)
tf_y = tf.placeholder(tf.float32, shape=y.shape)

# neural network layers
units = 10
hidden_1 = tf.layers.dense(tf_x, units, activation=tf.nn.relu)
output = tf.layers.dense(hidden_1, 1)

# define loss
loss = tf.losses.mean_squared_error(tf_y, output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	plt.ion()
	for step in range(200):
		# train net and ouput
		_, l, predict = sess.run([train_op, loss, output], feed_dict={tf_x:x, tf_y:y})
		# print(type(l))
		if step % 5 == 0:
			plt.cla()
			plt.scatter(x, y)
			plt.plot(x, predict, 'red', lw=5)
			plt.text(0.5, 0, 'loss=%.2f' %l, fontdict={'size':20, 'color':'blue'})
			plt.pause(0.1)

	plt.ioff()
	plt.show()