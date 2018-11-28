from __future__ import print_function

import numpy as np
import tensorflow as tf
import tflearn
import os
import sys

# Download the Titanic dataset
# from tflearn.datasets import titanic
# titanic.download_dataset('titanic_dataset.csv')



def breakBatch(data_and_target):
    data = []
    labels = []
    for i in range(len(data_and_target)):
        d = []
        for unit in data_and_target[i][1:]:
            d.append(unit)
        data.append(d)
        labels.append([1.0 - data_and_target[i][0], data_and_target[i][0]])
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32) 

# Preprocessing function
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
        # Converting 'sex' field to float (id is 1 after removing labels column)
        data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[1, 6]




train1_path = 'titanic_transformed_data.csv'
ITERATOR_BATCH_SIZE = 5
NR_EPOCHS = 3

dataset = tf.contrib.data.CsvDataset(train1_path,
                                     [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                                     header=False)


# data = preprocess([data], to_ignore)
# dataset = dataset.map(lambda x: tf.convert_to_tensor([preprocess([x], to_ignore)]))
# dataset = dataset.map(lambda x: x)
dataset = dataset.map(lambda *x: tf.convert_to_tensor(x))
dataset = dataset.repeat(20)
dataset = dataset.shuffle(100)
dataset = dataset.batch(ITERATOR_BATCH_SIZE)

# with tf.Session() as sess:
#     for i in range (NR_EPOCHS):
#         print('\nepoch: ', i)
#         iterator = dataset.make_one_shot_iterator()
#         next_element = iterator.get_next()
#         while True:            
#             try:
#               data_and_target = sess.run(next_element)
#             except tf.errors.OutOfRangeError:
#               break
#             print("\n\n", data_and_target)



# exit()

weightsFile = "model.ckpt"



# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)


# Preprocess data
data = preprocess(data, to_ignore)


# with open('titanic_transformed_data.csv', 'w') as f:
#     for i in range(len(data)):
#         f.write(str(labels[i][1]))
#         for j in range(len(data[i])):
#             f.write(',')
#             f.write(str(data[i][j]))
#         f.write('\n')

# exit()

# Build neural network
x = tf.placeholder(shape=(None, 6), dtype=tf.float32)
y = tf.placeholder(shape=(None, 2), dtype=tf.float32)

# net = tflearn.input_data(shape=[None, 6])
net = tflearn.input_data(placeholder=x)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
# model = tflearn.DNN(net)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=net, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)





init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()

# if os.path.exists(weightsFile + ".index"):
#     print("Loading checkpoint")
#     # model.load(weightsFile)
#     saver.restore(sess, os.path.join(os.getcwd(), weightsFile))



if len(sys.argv) > 1:
    NR_EPOCHS = int(sys.argv[1])

    # Start training (apply gradient descent algorithm)

    tflearn.is_training(True, session=sess)

    print(data)
    print(labels)

    # for i in range(10):
    #     sess.run(train_op, feed_dict={x: data, y: labels})
    #     cost = sess.run(loss, feed_dict={x: data, y: labels})

    #     print(str(cost))


    for i in range (NR_EPOCHS):
        print('\nepoch: ', i)

        if i != 0 and i % 10 == 0:
            # Let's create some data for DiCaprio and Winslet
            dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
            winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
            # Preprocess data
            dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)

            print('\n\n\n')

            pred = sess.run(net, feed_dict={x: [dicaprio, winslet]})
            # print(pred)

            # Predict surviving chances (class 1 results)
            # pred = model.predict([dicaprio, winslet])
            print("DiCaprio Surviving Rate:", '{:.5f}'.format(pred[0][1]))
            print("Winslet Surviving Rate:", '{:.5f}'.format(pred[1][1]))

            saver.save(sess, os.path.join(os.getcwd(), weightsFile))

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        cost = -1
        while True:            
            try:
                data_and_target = sess.run(next_element)
                data, labels = breakBatch(data_and_target)
                
                # print(data)
                # print(labels)
                # exit()

                sess.run(train_op, feed_dict={x: data, y: labels})
                cost = sess.run(loss, feed_dict={x: data, y: labels})
            except tf.errors.OutOfRangeError:
                break
            # print(data)
            # print(labels)

            # print('\n\n')

        print(str(cost))



    saver.save(sess, os.path.join(os.getcwd(), weightsFile))


    # model.fit(data, labels, n_epoch=epoch, batch_size=16, show_metric=True)

    # model.save(weightsFile)


# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)

print('\n\n\n')

pred = sess.run(net, feed_dict={x: [dicaprio, winslet]})
# print(pred)

# Predict surviving chances (class 1 results)
# pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", '{:.5f}'.format(pred[0][1]))
print("Winslet Surviving Rate:", '{:.5f}'.format(pred[1][1]))
