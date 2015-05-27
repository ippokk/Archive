# -*- coding: utf-8 -*-
import cv2
 
if __name__ == '__main__':
    # 画像の読み込み
    im = cv2.imread("00002100.jpg")
    # HoG特徴量の計算
    hog = cv2.HOGDescriptor()
    # SVMによる人検出
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
    # 人を検出した座標
    human, r = hog.detectMultiScale(im, **hogParams)
    # 長方形で人を囲う
    for (x, y, w, h) in human:
        cv2.rectangle(im, (x, y),(x+w, y+h),(0,50,255), 3)
    # 人を検出した座標
    cv2.imshow("Human detection",im)
    cv2.waitKey(0)
    # 画像保存
    # cv2.imwrite('test2.jpg',im)
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     # 画像の読み込み
#     im = cv2.imread("00002100.jpg")
#     # 探索用の機械学習ファイルを取得
#     cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")
#     # 探索(画像,縮小スケール,最低矩形数)
#     eye = cascade.detectMultiScale(im, 1.1, 3)
 
#     # 検出した部分を長方形で囲う
#     for (x, y, w, h) in eye:
#         cv2.rectangle(im, (x, y),(x+w, y+h),(0, 50, 255), 3)
 
#     # 画像表示
#     cv2.imshow("Show Image",im)
#     # キー入力待機
#     cv2.waitKey(0)
#     # 画像保存
#     cv2.imwrite("test2.jpg",im)
#     cv2.destroyAllWindows()