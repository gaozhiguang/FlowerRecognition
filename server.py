import tornado.ioloop
import tornado.web
import tornado.websocket
import numpy as np
import json
import os.path
import recognize_flowers
from PIL import Image
import matplotlib.pyplot as plt
import base64
import matplotlib.pyplot as plt

#/路径处理程序
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index6.html")

#/ws/路径处理程序
class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print("-------------[WebSocket Opened!]------------")

    def on_message(self, message):
        #print(message)
        #message是形如"data:image/gif;base64,R0lGODlhAwADAIAAAP/......”的格式，我们取逗号之后的部分
        value=message.split(',')
        tem=value[1]#tem是base64编码，是一个字符串
        img = base64.b64decode(tem)
        fh = open("temp.jpg","wb")
        fh.write(img)
        fh.close()
        
        '''
        # 从JSON字符串读取图片数据
        jsonObj=json.loads(message)
        data=[]
        for value in jsonObj["imgData"].values():
            data.append(value)
        print(len(data))#357604  等于299*299*4

        # 生成RGBA格式的Image对象，并保存本地
        arr=np.asarray(data,dtype=np.uint8)
        arr=np.reshape(arr,(299,299,4))
        img=Image.fromarray(arr,"RGBA")
        img.save("temp.png")
        '''

        img = Image.open("temp.jpg")
        plt.figure("Image") # 图像窗口名称
        plt.imshow(img)
        plt.axis('on') # 关掉坐标轴为 off
        plt.title('image') # 图像题目
        plt.show()


        # 调用识别函数
        print("==================识别开始==================")
        result = recognize_flowers.recognize_flowers("temp.jpg")
        #result="flower"
        print("==================识别结束==================")
        # 将结果返回网页
        self.write_message(result)

    def on_close(self):
        print("-------------[WebSocket Closed!]------------")

handlers=[
    (r"/",IndexHandler),
    (r"/connect",WSHandler)
]

if __name__=="__main__":
    app=tornado.web.Application(handlers,static_path=os.path.join(os.path.dirname(__file__),"static"),debug=True)
    app.listen(8000)
    tornado.ioloop.IOLoop.current().start()