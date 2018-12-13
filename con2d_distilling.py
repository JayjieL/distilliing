"""
mnist2--加载frozen.pb
"""
'''
对mnist集的初步训练，tensorflow神经网络的初使用
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# MNIST数据集相关常数
INPUT_NODE = 784
OUTPUT_NODE = 10


# 神经网络参数
LAYER1_NODE = 500
LAYER2_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8        # 基础学习率
LEARNING_RATE_DECAY = 0.99      # 学习率的衰减率
REGULARIZATION_RATE = 0.0001    #
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减率
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)


# 计算前向传播结果
def inference(input_tensor, avg_class, weight1, biases1, weight2, biases2,weight3, biases3, keep_prob):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biases1)
        layer1 = tf.nn.dropout(layer1, keep_prob,'layer1')
        layer2 = tf.nn.relu(tf.matmul(layer1, weight2) + biases2)
        layer2 = tf.nn.dropout(layer2, keep_prob,'layer2')
        return tf.matmul(layer2, weight3) + biases3

    else:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weight1))+avg_class.average(biases1))
        layer2 = tf.nn.relu(
            tf.matmul(layer1, avg_class.average(weight2))+avg_class.average(biases2))
        return tf.matmul(layer2, avg_class.average(weight3)) + avg_class.average(biases3)


# 训练模型的过程
x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y-input')
y_soft_target = tf.placeholder(tf.float32, [None, 10], name='sf_target')
#keep_prob = tf.placeholder(tf.float32, name='keep_prob')
T = tf.placeholder(tf.float32, name='tempalate')


# 生成隐藏层参数
weight1 = tf.Variable(
    tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
weight2 = tf.Variable(
    tf.truncated_normal([LAYER1_NODE, LAYER2_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[LAYER2_NODE]))

# 生成输出层参数
weight3 = tf.Variable(
    tf.truncated_normal([LAYER2_NODE, OUTPUT_NODE], stddev=0.1))
biases3 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

# 计算当前参数下神经网络的前向传播结果
layer1 = tf.nn.relu(tf.matmul(x, weight1) + biases1)
#layer1_drop = tf.nn.dropout(layer1, keep_prob,'layer1')
layer2 = tf.nn.relu(tf.matmul(layer1, weight2) + biases2) 
#layer2_drop = tf.nn.dropout(layer2, keep_prob,'layer2')
y =  tf.matmul(layer2, weight3) + biases3
#y = inference(x, None, weight1, biases1, weight2, biases2, weight3, biases3, keep_prob)

# 存储训练轮数的变量
global_step = tf.Variable(0, trainable=False)

# 计算交叉熵作为预测值与真实值之间差距的损失函数
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss1 = cross_entropy_mean
# 计算l2正则化损失函数
#regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
#regularization = regularizer(weight1)+regularizer(weight2)+regularizer(weight3)

#loss = cross_entropy_mean + regularization
hard_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1), name='hd_loss'))
soft_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_soft_target, name='sf_loss'))
alpha = 0.1
loss2 = hard_loss * alpha + soft_loss *(1-alpha) * tf.pow(T,2)

# 设置学习率
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE, global_step,
    mnist.train.num_examples/BATCH_SIZE, MOVING_AVERAGE_DECAY)

# 优化损失函数
train_step1 = tf.train.GradientDescentOptimizer(0.01).minimize(loss1, global_step=global_step)
train_step2 = tf.train.AdamOptimizer(learning_rate).minimize(loss2, global_step=global_step)
with tf.variable_scope('Accuracy'):
    #predictions = average_y
    predictions = tf.greater(y, 0, name="predictions")
    #correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(y_,1))
    correct_predictions = tf.equal(predictions, tf.cast(y_, tf.bool), name="correct_predictions")
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


"""
加载froze.pb
"""
import argparse   
import tensorflow as tf  
  
def load_graph(frozen_graph_filename):  
    # We parse the graph_def file  
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read())  
  
    # We load the graph_def in the default graph  
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name="prefix")

        #tf.import_graph_def(  
        #    graph_def,   
        #    input_map=None,   
         #   return_elements=None,   
         #   name="prefix",   
         #   op_dict=None,   
         #   producer_op_list=None  )  
    return graph  
  
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--frozen_model_filename", default="results_2/frozen_model_conv2d.pb", type=str, help="Frozen model file to import")  
    args = parser.parse_args()  
    #加载已经将参数固化后的图  
    graph = load_graph("results_2/frozen_model_conv2d.pb")  
    sess_teacher = tf.Session(graph=graph)
    # pred = sess_teacher.run(y_load, feed_dict={x_input:mnist.test.images,keep_prob:1.0})
    # We can list operations  
    #op.values() gives you a list of tensors it produces  
    #op.name gives you the name  
    #输入,输出结点也是operation,所以,我们可以得到operation的名字  
    #for op in graph.get_operations():  
    #    print(op.name,op.values())  
        # prefix/Placeholder/inputs_placeholder  
        # ...  
        # prefix/Accuracy/predictions  
    #操作有:prefix/Placeholder/inputs_placeholder  
    #操作有:prefix/Accuracy/predictions  
    #为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字  
    #注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字  
    x_input = graph.get_tensor_by_name('prefix/Placeholder/x-input:0')  
    keep_prob = graph.get_tensor_by_name('prefix/Placeholder/keep_prob:0')
    y_load = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')
    pred = sess_teacher.run(y_load, feed_dict={x_input:mnist.test.images,keep_prob:1.0})
    pred_np = np.argmax(pred,1)
    target = np.argmax(mnist.test.labels, 1)
    correct_prediction = np.sum(pred_np == target)
    print("teacher network accurary = ",correct_prediction /target.shape[0])
    params_t1 = 10 
    #params_t2 = 1      
    with tf.Session() as sess_student:
        sess_student.run(tf.global_variables_initializer())
        for i in range(10000):
            batch = mnist.train.next_batch(100)
            # teacher soft_target    
            soft_target = sess_teacher.run(y_load, feed_dict={x_input: batch[0], keep_prob:1.0})
            soft_target = tf.nn.softmax(soft_target/params_t1)
        
            # student train processing
            train_step2.run(feed_dict={x :batch[0], y_: batch[1], T : params_t1, y_soft_target:soft_target.eval() })
     
            if i % 200 == 0:
                hd_loss, sf_loss, loss_num, train_accuracy = sess_student.run([hard_loss, soft_loss ,loss2, accuracy], feed_dict={x:batch[0],  y_:batch[1],T:1.0, y_soft_target:soft_target.eval() }) 
                print('step %d, training accuracy %g , loss = %g , hard_loss = %g, soft_loss = %g' % (i, train_accuracy, loss_num, hd_loss, sf_loss ))
            if i % 1000 == 0:
                print('test accuracy %g' % sess_student.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels, T: 1.0,keep_prob:1.0}))
            
        print('Finally - test accuracy %g' % sess_student.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels, T: 1.0, krrp_prob:1.0}))


        











    



    
