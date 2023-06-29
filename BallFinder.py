from pathlib import Path
from typing import Any, Callable, Generator, Union
from torch import Tensor
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results, Boxes
from PIL import Image, ImageDraw, ImageFont


class BallFinder(YOLO):
    '''
    Ball Finder
    '''
    def __init__(self, model: Union[str, Path] = 'ball_finder.pt', task=None) -> None:
        super().__init__(model, task)

    # 通过外切矩形计算内切矩形
    @staticmethod
    def inner_square_coords(tup: tuple[float, float, float, float]):
        '''
        通过矩形的内切类圆计算内部矩形
        '''
        x0, y0, x1, y1 = tup
        # 计算外切正方形的边长
        side_length = x1 - x0

        # 计算需要缩短的边的长度
        delta = side_length * (1 - 1 / 2**0.5) / 2

        # 计算内切正方形的坐标
        inner_x0 = x0 + delta
        inner_y0 = y0 + delta
        inner_x1 = x1 - delta
        inner_y1 = y1 - delta

        return (inner_x0, inner_y0, inner_x1, inner_y1)

    def find(self, image: Image.Image, min_conf:float=0.6) -> Generator[tuple[tuple[int,int,int,int], float], Any, None]:
        results: list[Results] = self.predict(source=image)
        for result in results:
            boxes = result.boxes 
            if isinstance(boxes, Boxes):
                xyxys = boxes.xyxy
                confs = boxes.conf
                if isinstance(xyxys, Tensor) and isinstance(confs, Tensor):
                    for space,conf in zip(xyxys.cpu().numpy(), confs.cpu().numpy()):
                        if conf>=min_conf:
                            yield space,conf

    font = ImageFont.load_default()

    @staticmethod
    def draw_once(draw: ImageDraw.ImageDraw, space: tuple[int, int, int, int], text:str):
        draw.rectangle(space, outline='blue', width=2)
        draw.text( 
            xy=(space[0],space[1]), 
            text= text,
            font= BallFinder.font,
            fill= "blue",
        )

    def draw(self, image: Image.Image, min_conf:float=0.6) -> Image.Image:
        newimpage = image.copy()
        draw = ImageDraw.Draw(newimpage)
        for space, conf in self.find(image, min_conf):
            BallFinder.draw_once(draw, space, f"{conf:.3f}")
        return newimpage

    def crops(self, image: Image.Image) -> list[tuple[Image.Image, float, tuple[int, int, int, int]]]:
        return [(image.crop(space),conf,space) for space,conf in self.find(image)]


if __name__ == '__main__':
    # import time
    # finder = BallFinder(
    #     '/home/mole/projects/python/yolo/ball.v6i.yolov8/model/best.pt')
    # image = Image.open(
    #     '/home/mole/projects/python/yolo/ball.v4i.yolov8/balls/微信图片_20230609174346.jpg')
    # for crop,conf in finder.crops(image):
    #     crop.save(f'./crop_{time.time()}_{conf}.jpg')
    from pathlib import Path
    forder = Path('/home/mole/projects/python/yolo/SS_CombDet/datasets')
    finder = BallFinder('/home/mole/projects/python/yolo/SS_CombDet/ball_finder.pt')
    count = 0
    for file in forder.iterdir():
        if file.name.endswith('.jpg'):
            words = file.name.split('_')
            colors = [words[-1].split('.')[0], words[-3], words[-4].split('#')[0]]
            if len(set(colors)) != 1:
                raise ValueError(f"diff color in file name, file: {file.name}")
            color = colors[0]

            image = Image.open(file)
            crops = finder.crops(image)
            if color == 'white':
                if len(crops)!=0:
                    raise ValueError(f"white pic find something, filename: {file.name}, crops: f{crops}")
            else:
                if len(crops)!=1:
                    raise TypeError(f"find more than one ball, filename: {file.name}")
            for crop_image, crop_conf, space in crops:
                crop_image.save(file.parent/'crops'/file.name)
            # if len(crops)==0:continue
            # if len(crops)!=1:raise TypeError(f"find more than one ball, filename: {file.name}")
            # for crop_image, crop_conf in crops:
            #     crop_image.save(file.parent/'crops'/file.name)
