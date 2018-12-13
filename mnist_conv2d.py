'''
对mnist-卷积神经网络cumbersome_model
'''
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data


# MNIST数据集相关常数
INPUT_NODE = 784
OUTPUT_NODE = 10

MODEL_SAVE_PATH = "mnist_conv_model"
MODEL_NAME = "model.ckpt"
# 神经网络参数
LAYER1_NODE = 1200
LAYER2_NODE = 1200
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8        # 基础学习率
LEARNING_RATE_DECAY = 0.99      # 学习率的衰减率
REGULARIZATION_RATE = 0.0001    #
TRAINING_STEPS = 100000
MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减率


def weight_variable(shape,name):
    intial= tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(intial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x,W,name):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME',name=name)

def bias(conv, biases):
    return tf.nn.bias_add(conv, biases)

def max_pool_2x2(x,name):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)

# 训练模型的过程
def train(mnist):
    with tf.variable_scope('Placeholder'):
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y-input')
        x_image = tf.reshape(x,[-1,28,28,1])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # 生成隐藏层参数
    with tf.variable_scope('NN'):
        # First Layer
        W_conv1 = weight_variable([5,5,1,32],'W_conv1') 
        b_conv1 = bias_variable([32],'b_conv1')
        h_conv1 = tf.nn.relu(bias(conv2d(x_image,W_conv1,'h_conv1_xw') , b_conv1), name='h_conv1')
        h_pool1 = max_pool_2x2(h_conv1, 'h_pool1')

        # Second Layer
        W_conv2 = weight_variable([5,5,32,64], 'W_conv2')
        b_conv2 = bias_variable([64],'b_conv2')
        h_conv2 = tf.nn.relu(bias(conv2d(h_pool1, W_conv2,'h_conv2_xw') , b_conv2),name='h_conv2')
        h_pool2 = max_pool_2x2(h_conv2,'h_pool2')

        # Full Connected Layer1,上一层的输出为7*7*64的矩阵
        W_fc1 = weight_variable([7*7*64, 1024],'W_fc1')
        b_fc1 = bias_variable([1024],'b_fc1')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name='h_pool2_flat')
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='h_fc1')      
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

        # Full Connected Layer1
        W_fc2 = weight_variable([1024, 10],'W_fc2')
        b_fc2 = bias_variable([10],'b_fc2')
        y_conv = tf.matmul(h_fc1_drop, W_fc2,name='y_conv') + b_fc2  # 输出logits
    
    # 存储训练轮数的变量
    global_step = tf.Variable(0, trainable=False)

    # 计算交叉熵
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    #cross_entropy_mean = tf.reduce_mean(cross_entropy)
    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv,name='softmax_with_logits'),name='cross_entropy')

    # 计算l2正则化损失函数
    #regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #regularization = regularizer(weight1)+regularizer(weight2)+regularizer(weight3)

    loss = cross_entropy#_mean+regularization

    # 设置学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step,
        mnist.train.num_examples/BATCH_SIZE, MOVING_AVERAGE_DECAY)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss, global_step=global_step)#

    # 反向传播更新神经网络参数，并更新每个参数的滑动平均值
    #with tf.control_dependencies([train_step, variable_averages_op]):
    #    train_op = tf.no_op(name = 'train')

    with tf.variable_scope('Accuracy'):
        predictions = tf.greater(y_conv, 0, name="predictions")
        correct_predictions =tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
        
        #correct_predictions = tf.equal(predictions, tf.cast(y_, tf.bool), name="correct_predictions")
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32),name='accuracy')
    saver = tf.train.Saver(max_to_keep=15)
    # 初始化训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels, keep_prob:1}

        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels, keep_prob:1}

        # 迭代训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("After %d training step(s),validation accuracy"
                      " using average model is %g ,test accuracy using average model is %g" % (i, validate_acc, test_acc))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: xs, y_: ys, keep_prob:0.5})


        # 训练结束后，在测试集上检测最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s),test accuracy"
                      " using average model is %g" % (TRAINING_STEPS, test_acc))

        #在session当中就要将模型进行保存  
        #saver = tf.train.Saver()  
        #last_chkp = saver.save(sess, 'results_2/graph.chkp')
    #for op in tf.get_default_graph().get_operations():  
    #    print(op.name)  
# 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
    train(mnist)


# tensorflow提供的主程序入口
if __name__ == '__main__':
    tf.app.run()

        











    



    
