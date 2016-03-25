import tensorflow as tf
import numpy as np
import glob
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()

NoBall = []
Ball = []
#Import No Ball images
for filename_NoBall in glob.glob ('NaoCameraDataset/lower/img*.bmp'):
    img_data_NoBall = np.asarray(Image.open(filename_NoBall)).ravel()
    #~ x_NoBall = img_data_NoBall.ravel()
    NoBall.append(img_data_NoBall)

#Import ball images
for filename_Ball in glob.glob ('NaoCameraDataset/lower/Ball/img*.bmp'):
    img_data_Ball = np.asarray(Image.open(filename_Ball)).ravel()
    #~ x_Ball = img_data_Ball.ravel()
    Ball.append(img_data_Ball)

print(np.shape(Ball))
print(np.shape(NoBall))
x = np.concatenate((NoBall,Ball), axis = 0)
x = np.asarray(x)
xdot=x
#~ xdot = StandardScaler().fit_transform(x)
y_NoBall = np.zeros(622)
y_Ball = np.ones(342) 
ydot = np.concatenate((y_NoBall,y_Ball),axis= 0)


train_test = StratifiedShuffleSplit(ydot, n_iter=1, train_size = 700  , test_size = 264)
for train_validate_index, test_index in train_test:
	X_train_Validate, X_test = xdot[train_validate_index], xdot[test_index]
	y_train_Validate, y_test = ydot[train_validate_index], ydot[test_index]

#Placeholder for the input tensor
x = tf.placeholder(tf.float32, shape=[240*320*3],None)
y_ = tf.placeholder(tf.float32, shape=[2],None)

#Run session
sess.run(tf.initialize_all_variables())

#weight intialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#define convolution and pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# first layer
W_conv1 = weight_variable([5, 5, 3, 12])
b_conv1 = bias_variable([12])

x_image = tf.reshape(x, [-1,320,240,3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second layer
W_conv2 = weight_variable([5, 5, 12, 24])
b_conv2 = bias_variable([24])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# third
W_fc1 = weight_variable([80 * 60 * 24, 100])
b_fc1 = bias_variable([100])

h_pool2_flat = tf.reshape(h_pool2, [-1, 80*60*24])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax
W_fc2 = weight_variable([100, 1])
b_fc2 = bias_variable([1])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# actual computation
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,0), tf.argmax(y_,0))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print(np.shape(X_train_Validate))
print(np.shape(y_train_Validate))
def get_batch(n):
	train_test = StratifiedShuffleSplit(y_train_Validate, n_iter=1, train_size = n, test_size = 700-n)
	for x_batch, test_index in train_test:
		X_train_batch = X_train_Validate[x_batch]
		y_train_batch = y_train_Validate[x_batch]
	print(np.shape(X_train_batch))
	return X_train_batch, y_train_batch
  
for i in range(100):
	batch_x, batch_y = get_batch(50)
	print(np.shape(batch_y))
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_: batch_y, keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: X_test, y_: y_test, keep_prob: 1.0}))

