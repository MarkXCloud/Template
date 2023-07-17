from abc import ABCMeta, abstractmethod
import os
import numpy as np
import cv2


class _Visualizer(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def draw(self, image, label):
        pass


class ImageClassificationVisualizer(_Visualizer):
    def __init__(self, font_size=5, font_color=(255, 255, 255)):
        super().__init__()
        self.font_size = font_size
        self.font_color = font_color
        self.position = (224 // 2, 224 - 10)

    def draw(self, image: np.ndarray, label: str, img_name='result', root='./result'):
        print(f'prediction result: {label}')
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = cv2.putText(image, label, org=self.position, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=self.font_size,
                            color=self.font_color)
        img_name = img_name + '.jpg'
        cv2.imwrite(filename=os.path.join(root, img_name), img=image)
