'''
参考 https://blog.csdn.net/u010099080/article/details/52912439 
当使用如下代码保存使用 plt.savefig 保存生成的图片时，结果打开生成的图片却是一片空白。
plt.show()
plt.savefig("filename.png")
其实产生这个现象的原因很简单：在 plt.show() 后调用了 plt.savefig() ，
在 plt.show() 后实际上已经创建了一个新的空白的图片（坐标轴），
这时候你再 plt.savefig() 就会保存这个新生成的空白图片。
'''
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
import os
#全局数据定义
#flowernames=['Rose','Chinese rose','Oxalis','Nerium oleander','Marigold']
flowernames=['Oxalis','Nerium oleander','Marigold']
sourcepath='C:\\Users\\zhigu\\Desktop\\Comprehensive Project Practice\\flower-recognize\\recognize-flowers\\flower_photos'


#图片增强，增加图片数量
def image_strength(spath,flowername,filename):
        inimagepath=os.path.join(spath,flowername,filename)
        image_raw_data = tf.gfile.FastGFile(inimagepath, 'rb').read()
        with tf.Session() as sess:    
                img_data = tf.image.decode_jpeg(image_raw_data)    
                #plt.imshow(img_data.eval())    
                #plt.show()

                #上下翻转
                flipped = tf.image.flip_up_down(img_data)
                #plt.imshow(flipped.eval())
                #将图片进行重新编码
                encode_image = tf.image.encode_jpeg(flipped)
                #将编码后的数组变成我们能够参与计算的数组
                new_img = encode_image.eval() 
                filename1='updown'+filename
                outimagepath=os.path.join(spath,flowername,filename1)
                image = tf.gfile.FastGFile(outimagepath, "wb")
                image.write(new_img)
                #plt.savefig(outimagepath)
                #plt.show()

                #左右翻转
                flipped = tf.image.flip_left_right(img_data)
                #将图片进行重新编码
                encode_image = tf.image.encode_jpeg(flipped)
                #将编码后的数组变成我们能够参与计算的数组
                new_img = encode_image.eval() 
                #plt.imshow(flipped.eval())
                filename2='leftright'+filename
                outimagepath=os.path.join(spath,flowername,filename2)
                image = tf.gfile.FastGFile(outimagepath, "wb")
                image.write(new_img)
                #plt.savefig(outimagepath)
                #plt.show()

                #对角线翻转
                transposed = tf.image.transpose_image(img_data)
                #将图片进行重新编码
                encode_image = tf.image.encode_jpeg(transposed)
                #将编码后的数组变成我们能够参与计算的数组
                new_img = encode_image.eval() 
                #plt.imshow(transposed.eval())
                filename3='transpose'+filename
                outimagepath=os.path.join(spath,flowername,filename3)
                image = tf.gfile.FastGFile(outimagepath, "wb")
                image.write(new_img)
                #plt.savefig(outimagepath)
                #plt.show()

                #调整图像亮度、饱和度、色相函数,亮度饱和度和色相随机调整
                # 在[-max_delta, max_delta)的范围随机调整图片的亮度。
                adjusted = tf.image.random_brightness(img_data, max_delta=0.2)#亮度
                # 在[lower, upper]的范围随机调整图的对比度。
                #adjusted = tf.image.random_contrast(adjusted, 0.1, 0.6)
                # 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间。
                adjusted = tf.image.random_hue(adjusted, 0.1)
                # 在[lower, upper]的范围随机调整图的饱和度。
                adjusted = tf.image.random_saturation(adjusted, 0, 1)
                #将图片进行重新编码
                encode_image = tf.image.encode_jpeg(adjusted)
                #将编码后的数组变成我们能够参与计算的数组
                new_img = encode_image.eval() 
                #plt.imshow(adjusted.eval())
                filename4='light'+filename
                outimagepath=os.path.join(spath,flowername,filename4)
                image = tf.gfile.FastGFile(outimagepath, "wb")
                image.write(new_img)
                #plt.savefig(outimagepath)
                #plt.show()


    

for sub_dir in flowernames:
    temppath=os.path.join(sourcepath,sub_dir)
    parents = os.listdir(temppath)
    for filename in parents:
        image_strength(sourcepath,sub_dir,filename)


'''
<input type="file" capture="camera" accept="image/*" id="filetest" name="filetest">
'''