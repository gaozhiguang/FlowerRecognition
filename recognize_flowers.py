import os, sys
import data_process
import train
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

#数据集路径
#INPUT_DATA = r'D:\deep-learning\recognize-flowers\flower_processed_data.npy'
INPUT_DATA = 'flower_processed_data.npy'
#训练好模型路径
#MODEL_PATH = r'D:\deep-learning\recognize-flowers\train_dir\model.pb'
MODEL_PATH = './train_dir/model.pb'
#获取标签集
processed_data = np.load(INPUT_DATA)
#获取数据标签
label_lines=processed_data[6] 
#print("label_lines:",label_lines)
#设置log级别，只显示warning和error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#测试图片路径
#IMAGE_PATH = r'D:\deep-learning\recognize-flowers\source_photos\万寿菊\3 (1).jpg'
IMAGE_PATH =r'C:\Users\zhigu\Desktop\Comprehensive Project Practice\flower-recognize\recognize-flowers\flower_photos\Chinese rose\IMG_20190518_110249.jpg'


def recognize_flowers(IMAGE_PATH):
    '''
        函数名称：recognize_flowers()
        参数含义：IMAGE_PATH：识别图片的路径
        函数返回：返回预测的结果
    '''
    #加载已训练好的inception-v3模型处理图片数据
    graph, bottleneck_tensor, jpeg_data_tensor =data_process.create_inception_graph()
    with tf.Session(graph=graph) as sess:
        #初始化会话
        init = tf.global_variables_initializer()
        sess.run(init)
        #读取图片
        image_raw_data = gfile.FastGFile(IMAGE_PATH, 'rb').read()
        #通过inception-v3模型处理图片获得瓶颈层的图片数据
        image_value = sess.run(bottleneck_tensor,{jpeg_data_tensor:image_raw_data})
        #将结果转换成一维数据
        image_value = np.squeeze(image_value)

    #加载训练好的模型进行花卉识别
    with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
        # 新建GraphDef文件，用于临时载入模型中的图
        graph_def = tf.GraphDef()
        # GraphDef加载模型中的图
        graph_def.ParseFromString(f.read())
        #将graph_def导入当前默认的Graph(在空白图中加载GraphDef中的图)
        tf.import_graph_def(graph_def, name='')
        with tf.Session() as sess:
            #输入图片并且获取图片的结果
            init = tf.global_variables_initializer()
            sess.run(init)
            #在图中获取张量需要使用graph.get_tensor_by_name加张量名
            softmax_tensor = sess.graph.get_tensor_by_name('output/prob:0')
            #获取预测结果
            predictions = sess.run(softmax_tensor, {'BottleneckInputPlaceholder:0': [image_value]})
            #将预测结果转变成一维数组(从数组的形状中删除单维度条目，即把shape中为1的维度去掉)
            #predictions=np.squeeze(predictions)
            print("predictions:",predictions)
            print("predictions[0]:",predictions[0])
            #按置信区间顺序排序以显示第一个预测的标签
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            print("top_k:",top_k)
            #获取最有可能的识别结果
            human_string = label_lines[top_k[0]]
            print("human_string:",human_string)
            #将所有结果和可能性输出
            # for node_id in top_k:
            #     human_string = label_lines[node_id]
            #     score = predictions[0][node_id]
            #     print('%s (score = %.5f)' % (human_string, score))
            return human_string
#recognize_flowers(IMAGE_PATH)