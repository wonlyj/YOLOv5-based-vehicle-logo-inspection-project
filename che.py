""""""
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device
import cv2
import mss
import pyautogui
import numpy as np




# 获取当前屏幕得分辨率
mx , my = pyautogui.size()
print( mx, my )

# 屏幕得中心点
cx = int(mx//2)  # x
cy = int(my//2) # y
# 获取 640x640得左上角    通过中心点计算
dlx = cx - 320
dly = cy - 320
# 获取 640x640得右下角   通过中心点计算
drx = cx + 320
dry = cy + 320
# 小图得中心点，这个中心点虽然和屏幕中心点重叠，
# 但是不能和屏幕中心点xy一样，要按照小图计算中心点
dcx = 320
dcy = 320
# 定义mss实例
m = mss.mss()
# 设定检测设备  这里如果想要使用 显卡进行识别检测，将cpu更改为 0 即可
device = 'cpu'
# 查找设备
device = select_device(device)
# 指定权重文件   这里是自训练的
weights = 'best.pt'
# 用于各种后台的Python推理  ，参数为默认参数
model = DetectMultiBackend(weights, device=device, dnn=False, data=False, fp16=False)
# 公共属性，控制鼠标移动的开关
is_right_clik = False




# 开始检测
def run():

    while True:
        global is_right_clik
        # mss风格 例如 x,y,w,h，左上角x和y， 宽度    高度
        monitor = dict(left=dlx,top=dly,width=640,height=640)
        mss_img = m.grab(monitor) #进行截图返回mss对象
        # 将mss对象转换成 numpy得数组 为了能通过opencv进行展示
        img = np.array( mss_img )
        # 去掉透明度
        img = img[:,:,:3]
        img = cv2.cvtColor( img, cv2.COLOR_BGRA2BGR)
        img_shape = img.shape # 获取维度
        # 检测  处理图片 ，此处理图片可以在 detect.py文件中找寻逻辑
        im = letterbox(img, (640,640), stride=45, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # 进行推理
        pred = model(im, augment=False, visualize=False)
        # 非最大抑制，是一种在计算机视觉和目标检测领域常用的技术，
        # 用于去除冗余的检测结果，保留最相关的目标
        # 参数默认值 可在detect.py文件中找到逻辑
        pred = non_max_suppression(pred, 0.4, 0.2, None, False, max_det=1000)[0]
        if len(pred):
            # 将框从img_size重新缩放为im0大小
            pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], img_shape).round()
            for *xyxy, conf, cls in reversed(pred): # 返回 torch.tenosr数据
                # 将 tensor数据提取出来  x1 y1 是左上角， x2y2是右下角
                x1,y1,x2,y2 = map(round, (torch.tensor(xyxy).view(1,4).view(-1).tolist()))
                # x和y是中心点   wh是宽高
                x,y,w,h = map( round, ((xyxy2xywh(torch.tensor(xyxy).view(1,4))).view(-1).tolist()))

                class_names = ['QiYa', 'AoDi', 'BieKe', 'BenTian', 'BiaoZhi','ChangAN' ,'DaZhong','FengTian','Jeep','MaZiDa','QiChen','RiChan','XueFuLan','XueTieLong']  # 示例类名列表


                # 获取类名
                label = class_names[int(cls)]
                print(label)
                # 在小图中画矩形7
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

                # 在框上方添加文本，这里我们将文本位置设置为框的左上角向上移动一定的像素
                text_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20)
                cv2.putText(img, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5,
                            cv2.LINE_AA)


        cv2.imshow('img', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break



if __name__ == '__main__':
    run()
