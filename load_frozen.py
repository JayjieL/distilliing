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
        tf.import_graph_def(  
            graph_def,   
            input_map=None,   
            return_elements=None,   
            name="prefix",   
            op_dict=None,   
            producer_op_list=None  
        )  
    return graph  
  
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str, help="Frozen model file to import")  
    args = parser.parse_args()  
    #加载已经将参数固化后的图  
    graph = load_graph(args.frozen_model_filename)  
  
    # We can list operations  
    #op.values() gives you a list of tensors it produces  
    #op.name gives you the name  
    #输入,输出结点也是operation,所以,我们可以得到operation的名字  
    for op in graph.get_operations():  
        print(op.name,op.values())  
        # prefix/Placeholder/inputs_placeholder  
        # ...  
        # prefix/Accuracy/predictions  
    #操作有:prefix/Placeholder/inputs_placeholder  
    #操作有:prefix/Accuracy/predictions  
    #为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字  
    #注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字  
    x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')  
    y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')  
    params_t = 1      
    with tf.Session(graph=graph) as sess:
        
        for i in range(11000):
            batch = mnist.train.next_batch(100)
            # teacher soft_target    
            soft_target = sess_teacher.run(y, feed_dict={x_input: batch[0]})
            soft_target = tf.nn.softmax(soft_target/params_t)
        
            # student train processing
            train_step.run(feed_dict={x_ :batch[0], y_: batch[1], T : params_t, y_soft_target:soft_target.eval() })
            if i % 200 == 0:
                hd_loss, sf_loss, loss_num, train_accuracy = sess_student.run([hard_loss, soft_loss ,Loss, accuracy], feed_dict={x_:batch[0],  y_:batch[1],
                                                                   T:1.0, y_soft_target:soft_target.eval()  }) 
                print('step %d, training accuracy %g , loss = %g , hard_loss = %g, soft_loss = %g' % (i, train_accuracy, loss_num, hd_loss, sf_loss ))
            if i % 1000 == 0:
                print('test accuracy %g' % sess_student.run(accuracy,feed_dict={x_: mnist.test.images, y_: mnist.test.labels, T: 1.0}))
            
    print('Finally - test accuracy %g' % sess_student.run(accuracy,feed_dict={
                                    x_: mnist.test.images, y_: mnist.test.labels, T: 1.0}))

