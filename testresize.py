import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
import os

#按照比例裁剪图像，第二个参数为调整比例，比例取值[0,1]
#central_cropped = tf.image.central_crop(img_data,0.5)      
# 
# plt.imshow()函数负责对图像进行处理，并显示其格式，而plt.show()则是将plt.imshow()处理后的图像显示出来。          


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
sourcepath='G:\\flowers\\Camera'
targetpath='C:\\Users\\zhigu\\Pictures'
scaleimage(sourcepath,targetpath,'Rose','IMG_20190518_112117.jpg')