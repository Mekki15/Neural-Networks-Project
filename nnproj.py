import tensorflow as tf
import numpy as np
import glob
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit

sess = tf.InteractiveSession()

NoBall = []
Ball = []

# Import no-ball images
for filename_NoBall in glob.glob ('NaoCameraDataset/lower/img*.bmp'):
    img_data_NoBall = np.asarray(Image.open(filename_NoBall)).ravel()
    NoBall.append(img_data_NoBall)

# Import ball images
for filename_Ball in glob.glob ('NaoCameraDataset/lower/Ball/img*.bmp'):
    img_data_Ball = np.asarray(Image.open(filename_Ball)).ravel()
    Ball.append(img_data_Ball)

# Count images
num_no_ball = np.shape(NoBall)[0]
num_ball = np.shape(Ball)[0]

# Create dataset (input images + labels)
xdot = np.asarray(np.concatenate((NoBall, Ball), axis = 0))
xdot = StandardScaler().fit_transform(xdot) 		# Do we need this??

y_NoBall = np.concatenate((np.zeros((num_no_ball, 1)), np.ones((num_no_ball, 1))), axis=1)
y_Ball = np.concatenate((np.ones((num_ball, 1)), np.zeros((num_ball, 1))), axis=1)
print(np.shape(NoBall))
print(np.shape(y_Ball))
#~ y_Ball = np.ones(num_ball) 
ydot = np.concatenate((y_NoBall,y_Ball),axis= 0)
print(np.shape(ydot))

# Split into train set and test set 
train_test = StratifiedShuffleSplit(ydot[:, 0], train_size = 0.7)
for train_validate_index, test_index in train_test:
	X_train_Validate, X_test = xdot[train_validate_index], xdot[test_index]
	y_train_Validate, y_test = ydot[train_validate_index], ydot[test_index]

####
# Weight constructor
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# Bias constructor
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# Convolutional layer constructor
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Pooling layer constructor
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

####
# Input layer
x = tf.placeholder(tf.float32, shape=[None, 76800*3])

# Label reference
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# First conv layer
W_conv1 = weight_variable([12, 12, 3, 16])
b_conv1 = bias_variable([16])
x_image = tf.reshape(x, [-1,320,240,3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# First pool layer
h_pool1 = max_pool_2x2(h_conv1)

# Second conv layer
W_conv2 = weight_variable([12, 12, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# Second pool layer
h_pool2 = max_pool_2x2(h_conv2)

# First fully connected layer
W_fc1 = weight_variable([80 * 60 * 32, 512])
b_fc1 = bias_variable([512])
h_pool2_flat = tf.reshape(h_pool2, [-1, 80*60*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout (to avoid overfitting)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Second fully connected layer - Softmax
W_fc2 = weight_variable([512, 2])
b_fc2 = bias_variable([2])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

####
# Back-propagation
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize variables
sess.run(tf.initialize_all_variables())

#  Op to save the network
saver = tf.train.Saver()

# Extraxt a batch of random images from the training set
def get_batch(n):
	train_test = StratifiedShuffleSplit(y_train_Validate, n_iter=1, train_size = n)  # Change this!
	for x_batch, test_index in train_test:
		X_train_batch = X_train_Validate[x_batch]
		y_train_batch = y_train_Validate[x_batch]
	return X_train_batch, y_train_batch

# Training
for i in range(100):
	batch_x, batch_y = get_batch(50)
	#~ if i%100 == 0:
	train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
	print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

# Testing
print("test accuracy %g"%accuracy.eval(feed_dict={x: X_test, y_: y_test, keep_prob: 1.0}))

# Save the variables to disk
save_path = saver.save(sess, "model.ckpt")
print("Model saved in file: %s" % save_path)


