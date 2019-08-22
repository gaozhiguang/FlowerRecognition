# FlowerRecognition

花卉识别——基于Inception-V3模型

***
## 功能描述

用户在手机Web端启用拍照功能，拍一张花的图像，上传到服务器端，后端采用Inception-V3的学习迁移模型对花朵图像进行识别，把识别结果返回给手机Web端。目前支持“夹竹桃，
万寿菊，蔷薇，月季，红花酢浆草”五种花卉的识别，可以通过重新训练模型来支持其他花卉。

Inception-V3模型是谷歌在大型图像数据库ImageNet上训练好了的一个图像分类模型，这个模型可以对1000种类别的图片进行图像分类。但现成的Inception-V3无法对“花”类别图片做进一步细分，因此本花朵识别项目是在Inception-V3模型基础上采用迁移学习方式完成对“花”类别图片进一步细分。

## 运行环境

*后端*  
- Python
- Python库
  - tornado
  - tensorflow或tensorflow-gpu
  - 以上库可以在命令行界面用`pip install tornado`命令安装，tensorflow类似
  - pillow [安装教程](https://www.cnblogs.com/yuanzhoulvpi/p/9028713.html)

*前端*  
Internet Explorer 9、Firefox、Opera、Chrome 以及 Safari 浏览器

## 运行方法

*后端*  
Windows：按下win+r，输入`cmd`，回车，进入cmd命令行界面。输入`cd /d 你的代码存储目录`，进入代码存储目录，输入`python server.py`运行后端服务器

*重新训练模型*  
如需支持其他花卉，首先为每种花卉准备100张图片，分类存放在源码original_photos目录下，每种花卉一个文件夹，文件夹名为花卉名称。然后将numberEnhance.py文件中的flowerlist列表修改为需要的花卉名称。按下win+r，输入`cmd`，回车，进入cmd命令行界面。输入`cd /d 你的代码存储目录`，进入代码存储目录，依次用python命令执行numberEnhance.py、data_process.py、train.py三个文件。最后会在源码目录下生成一个flower_processed_data.npy文件，即为最终模型。

*前端*  
在浏览器地址栏输入`服务器IP:8100`，回车即可进入网页。按照提示调用相机拍照或选择图片文件，进行裁剪，点击确定，即可进行识别。
 
            
