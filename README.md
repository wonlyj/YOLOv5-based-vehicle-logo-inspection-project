# YOLOv5-based-vehicle-logo-inspection-project

1、在test.py文件中指定数据集配置文件和训练结果模型<br>
得到best.pt
<img width="978" alt="屏幕截图 2024-12-31 221357" src="https://github.com/user-attachments/assets/c2cb1380-457a-4b99-8493-9d7361212365" />

2、<br>
（1）运用LabelImg图像标注工具，给项目所需的数据集进行数据标签标注：① Open Dir：指定要标注的图像所在的目录；② Change Save Dir：指定标注数据保存的目录；③ Change Save Format：选定YOLO作为指定标签的保存格式。
（2）使用Yolov5制作并训练自己的数据集，得到车标识别模型。
（3）threading模块创建和管理线程，创建一个线程来监听键盘事件，实现了程序的并发执行。
（4）实现非极大值抑制算法以去除冗余的边界框，并确定最终的检测结果。这一步骤有助于提高检测结果的准确性和可靠性。
（5）使用OpenCV库在图像上绘制边界框和标签。我们可以根据检测到的车标位置和类别信息，在图像上绘制出相应大小和颜色的边界框，并在边界框旁边标注出汽车的品牌名称。
