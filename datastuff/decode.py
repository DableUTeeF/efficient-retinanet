import base64
import cv2
import json
from imageio import imread
import io


a = json.load(open('/media/palm/data/MicroAlgae/jsn/MOPH/OVlabel2/OV egg kato 40X (1625).json'))
# b64_bytes = base64.b64encode(a['imageData'])
# reconstruct image as an numpy array
img = imread(io.BytesIO(base64.b64decode(a['imageData'])))
print(a['shapes'][0]['points'][0])
print(a['shapes'][0]['points'][1])
img = cv2.rectangle(img,
                    (int(a['shapes'][0]['points'][0][0]), int(a['shapes'][0]['points'][0][1])),
                    (int(a['shapes'][0]['points'][1][0]), int(a['shapes'][0]['points'][1][1])),
                    (0, 0, 255),
                    thickness=10)
img = cv2.resize(img, (800, 800))
cv2.imshow('t', img)
cv2.waitKey()
