import tensorflow as tf

def f():
	with tf.variable_scope('A') as scope:
		print scope.reuse

f()
