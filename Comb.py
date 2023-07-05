import cv2
import torch 
from PIL import Image, ImageDraw

from BallFinder import BallFinder
from ColorClassifier import ColorClassifier

class CombineModel:
    _finder: BallFinder
    _classifier: ColorClassifier
    def __init__(self) -> None:
        self._finder = BallFinder()
        self._classifier = ColorClassifier()
    
    def find(self, image: Image.Image):
        return [self._classifier.classify(im) for im,_,_ in self._finder.crops(image)]

    def draw(self, image: Image.Image) -> Image.Image:
        newimage = image.copy()
        draw = ImageDraw.Draw(newimage)
        for im,conf,space in self._finder.crops(image):
            color = self._classifier.classify(im)
            BallFinder.draw_once(draw, space, f" {color}:{conf}")
        return newimage
    
    def process_rtmp_stream(self, rtmp_url):
        # 打开RTMP流
        stream = cv2.VideoCapture(rtmp_url)
        
        # 检查流是否打开
        if not stream.isOpened():
            print("Could not open stream")
            return
        
        # 循环处理视频的每一帧
        while True:
            ret, frame = stream.read()
            if not ret:
                break
            
            # 将帧转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 使用YOLO进行目标检测
            targets = self._finder.find(pil_image)
            
            # 遍历检测到的目标
            for xyxy,conf in targets:
                x1, y1, x2, y2 = xyxy
                # 裁剪目标图像
                target_image = pil_image.crop(xyxy)
                
                # 使用分类模型对目标进行分类
                class_label = self._classifier.classify(target_image)
                
                # 将分类标签添加到帧上
                cv2.putText(frame, class_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            # 显示帧
            cv2.imshow('frame', frame)
            
            # 按'q'键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放流和销毁窗口
        stream.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    model = CombineModel()
    model.process_rtmp_stream("rtmp://pili-live-rtmp.quqqi.com/sslive/K18-opc")