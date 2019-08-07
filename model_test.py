import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
from keras_retinanet.models.efficientnet import EfficientNetBackbone


if __name__ == '__main__':
    b0 = EfficientNetBackbone('efficientnetb0')
    b0.validate()
    b0_m = b0.retinanet(5)
    b0_m.compile('sgd', 'mse')
    b1 = EfficientNetBackbone('efficientnetb1')
    b1.validate()
    b1_m = b1.retinanet(5)
    b1_m.compile('sgd', 'mse')
    b2 = EfficientNetBackbone('efficientnetb2')
    b2.validate()
    b2_m = b2.retinanet(5)
    b2_m.compile('sgd', 'mse')
    b3 = EfficientNetBackbone('efficientnetb3')
    b3.validate()
    b3_m = b3.retinanet(5)
    b3_m.compile('sgd', 'mse')
    b4 = EfficientNetBackbone('efficientnetb4')
    b4.validate()
    b4_m = b4.retinanet(5)
    b4_m.compile('sgd', 'mse')
    b5 = EfficientNetBackbone('efficientnetb5')
    b5.validate()
    b5_m = b5.retinanet(5)
    b5_m.compile('sgd', 'mse')
