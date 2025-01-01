# YOLOv5-based-vehicle-logo-inspection-project

1、在test.py文件中指定数据集配置文件和训练结果模型,得到best.pt<br>
<img width="978" alt="屏幕截图 2024-12-31 221357" src="https://github.com/user-attachments/assets/c2cb1380-457a-4b99-8493-9d7361212365" /><br>
2、项目效果<br>
（1）运用LabelImg图像标注工具，给项目所需的数据集进行数据标签标注：① Open Dir：指定要标注的图像所在的目录；② Change Save Dir：指定标注数据保存的目录；③ Change Save Format：选定YOLO作为指定标签的保存格式。<br>
（2）使用Yolov5制作并训练自己的数据集，得到车标识别模型。<br>
（3）threading模块创建和管理线程，创建一个线程来监听键盘事件，实现了程序的并发执行。<br>
（4）实现非极大值抑制算法以去除冗余的边界框，并确定最终的检测结果。这一步骤有助于提高检测结果的准确性和可靠性。<br>
（5）使用OpenCV库在图像上绘制边界框和标签。我们可以根据检测到的车标位置和类别信息，在图像上绘制出相应大小和颜色的边界框，并在边界框旁边标注出汽车的品牌名称。<br>
3、实现步骤<br>
(1)、环境安装<br>
首先进入YOLOv5开源网址 ，手动下载zip，代码文件夹中有requirements.txt文件，里面描述了所需要的安装包。<br>
最终安装的pytorch版本为2.2.2，torchvision版本为0.17.2，python版本为3.11<br>
(2)、制作数据集<br>
① 准备数据集：<br>
收集包含各种车标的图像，并将它们分为训练集、验证集和测试集。拍摄图像应当多种角度，确保图像清晰以及数据鲁棒性。<br>
② 转换数据集格式：<br>
运用LabelImg图像标注工具，给项目所需的数据集进行数据标签标注。 <br>
<img width="1278" alt="1" src="https://github.com/user-attachments/assets/d2793b85-e6da-4813-8ff2-905441eac113" /><br>



