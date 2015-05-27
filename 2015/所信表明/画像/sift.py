# -*- coding: utf-8 -*-
import cv2
import numpy as np
 
def main():
    # 画像の取得
    im = cv2.imread("t3.JPG")
    # グレースケール変換
    gray = cv2.imread("t3.JPG", 0)
    # gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    # 特徴点の抽出
    kp = sift.detect(gray,None)
    # 特徴点に小さい円を描画
    im = cv2.drawKeypoints(im,kp)
    # 結果の保存
    cv2.imwrite("t3_sift.png",im)
 
if __name__ == '__main__':
    main()