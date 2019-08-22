import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
# 加载train.py中定义的常量和前向传播的函数
import train

# 下载的谷歌训练好的inception-v3模型文件名
MODEL_FILE = './inceptionV3/tensorflow_inception_graph.pb'
# inception-v3 模型中代表瓶颈层结果的张量名称
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
# 图像输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
model_file1 = "./train_dir/model.pb"
# inception-v3 模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

def create_inception_graph():
    #加载已训练好的inception-v3模型
    with tf.Graph().as_default() as graph:
        with gfile.FastGFile(MODEL_FILE, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, name='', return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
    return graph, bottleneck_tensor, jpeg_data_tensor

def flower_recog(picturename):

    # 定义喂数据占位张量
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder');
    ground_truth_input = tf.placeholder(tf.int64, [None], name='GroundTruthInput')


    graph, bottleneck_tensor, jpeg_data_tensor =create_inception_graph()
    correct_prediction = tf.argmax(final_tensor, 1)

    with tf.Session(graph=graph) as sess:

        # 读取并解析图片，将图片转化为299*299以方便inception-v3模型来处理。
        image_raw_data = gfile.FastGFile(picturename, 'rb').read()
        image_value = sess.run(bottleneck_tensor,{jpeg_data_tensor:image_raw_data})
        image_value = np.squeeze(image_value)

        init = tf.global_variables_initializer()
        sess.run(init)

    with tf.Session() as sess:
        # 初始化参数
        init = tf.global_variables_initializer()
        sess.run(init)
        
        sess.run(train_step, feed_dict={
                bottleneck_input: image_value[start: end]})

    return correct_prediction
print(flower_recog("temp.jpg"))