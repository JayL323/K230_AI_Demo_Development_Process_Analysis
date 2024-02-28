#coding = utf-8
import cv2
import numpy as np
from sklearn import preprocessing
from mobile_face_net import MobileFaceNet

# 1_affine.png、2_affine.png是同一个人，3_affine.png与前两者是不同人
img_lists = ['bin/1_affine.png','bin/2_affine.png','bin/3_affine.png']

face_recg = MobileFaceNet()

embeddings = []
for i,img_file in enumerate(img_lists):
    ori_img = cv2.imread(img_file)
    input_data = face_recg.pre_process(ori_img)
    embedding = face_recg.farward(input_data)
    embedding_norm = preprocessing.normalize(embedding)
    embeddings.append(embedding_norm)

embedding_one = embeddings[0]
scores = np.array([np.sum(embedding_one * emb_database) / 2 + 0.5 for emb_database in embeddings])
print("scores:",scores)





