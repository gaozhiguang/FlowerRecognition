'''
>>> import os
>>> path='G:\\flowers\\Camera\\Rose'
>>> filename='a.jpg'
>>> dir=os.path.join(path,filename)
>>> dir
'G:\\flowers\\Camera\\Rose\\a.jpg'
'''
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
import os
#定义的全局变量
sourcepath='G:\\flowers\\Camera'#这里不能写成'G:\flowers\Camera'
targetpath='C:\\Users\\zhigu\\Desktop\\Comprehensive Project Practice\\flower-recognize\\recognize-flowers\\flower_photos'
#四个参数的含义：spath：源路径，tpath：目标路径，flowername：花名（子文件夹），filename：图片名
def scaleimage(spath,tpath,flowername,filename):
    inimagefile=os.path.join(spath,flowername,filename)
    image_raw_data = tf.gfile.FastGFile(inimagefile, 'rb').read()
    with tf.Session() as sess:    
        img_data = tf.image.decode_jpeg(image_raw_data) 
        #图片缩放成299*299
        '''
        method=0：双线性插值法（Bilinear interpolation）；
        method=1：最近邻法（Nearest neighbor interpolation）；
        method=2：双三次插值法（Bicubic interpolation）；
        method=3：面积插值法（Area interpolation）。
        '''
        resized = tf.image.resize_images(img_data, [299, 299], method=0)
        #print(sess.run(tf.shape(resized)))#shape=3
        #resized = tf.cast(resized, tf.int32)
        resized = tf.cast(resized, tf.uint8)
        #将图片进行重新编码
        encode_image = tf.image.encode_jpeg(resized)
        #将编码后的数组变成我们能够参与计算的数组
        new_img = encode_image.eval() 
        #plt.imshow(resized.eval())
        outimagefile=os.path.join(tpath,flowername,filename)
        #plt.savefig(outimagefile)
        #plt.show()
        #将图片保存到目标目录下
        image = tf.gfile.FastGFile(outimagefile, "wb")
        image.write(new_img)


#获取五个子文件夹
#sub_dirs = [x[0] for x in os.walk(sourcepath)]
#print(len(sub_dirs))#结果显示为6
#print(sub_dirs)
'''
['G:\\flowers\\Camera', 'G:\\flowers\\Camera\\Chinese rose', 'G:\\flowers\\Camera\\Marigold', 
    'G:\\flowers\\Camera\\Nerium oleander', 'G:\\flowers\\Camera\\Oxalis', 'G:\\flowers\\Camera\\Rose']
'''
# 读取所有的子目录。
#for sub_dir in sub_dirs[1:]:
    # 获取一个子目录中所有的图片文件。
    #parents = os.listdir(sub_dir)
    #print(parents)#类似'IMG_20190518_113924.jpg', 'IMG_20190518_113930.jpg', 'IMG_20190518_113932.jpg'
    #print('\n')
    #print(parents[0])
    #print('\n')
    #print(len(parents))
flowernames=['Rose','Chinese rose','Oxalis','Nerium oleander','Marigold']
#flowernames=['Oxalis','Nerium oleander','Marigold']
for sub_dir in flowernames:
    temppath=os.path.join(sourcepath,sub_dir)
    parents = os.listdir(temppath)
    for filename in parents:
        scaleimage(sourcepath,targetpath,sub_dir,filename)