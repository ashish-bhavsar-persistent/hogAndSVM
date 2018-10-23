#SVMDetectorUsingHOGfeature

====

## Overview

自身の顔検出器の作成

## Description

main()において、
capture_path: captureした画像を保存するフォルダ
train_imgs_path: 訓練データを入れてある/入れるフォルダ。positive dataはpos-, negative dataはneg-がぞれぞれファイル名に入っていること。

save_train_features_path: 訓練データの特徴量を入れてある/入れるpath
save_train_label_path: 訓練データのラベルを入れてある/入れるpath
test_img_path: 一つだけテストしたい場合のテストデータの正解してほしいデータを入れてあるフォルダ。
dump_path: 学習した識別器のデータを保存するpath
train_pos_imgs_path: 訓練データの正解してほしいデータを入れてある/入れるフォルダ。
train_neg_imgs_path: 訓練データのnegative判定してほしいデータを入れてある/入れるフォルダ。

test_pos_imgs_path: テストデータの正解してほしいデータを入れてある/入れるフォルダ。
test_neg_imgs_path: テストデータのnegative判定してほしいデータを入れてある/入れるフォルダ。

num_of_max_sample: captureしたいサンプル数

を表しており、mainを実行すれば、train_dataの作成から、識別器の学習、テストまでを一貫して行える。
適宜関数をコメントアウトして、途中の過程をスキップできる。

HumanFaceDetect()を実行することで、様々な顔検出用の識別器をBest_humans_detector.pklに保存することができる。main()ではこれを呼び出し、自身の顔検出用の識別器を作成する。

## UsingDataset
使用したデータセットは、The Database of Faces
(http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)
->顔検出器の　train用の　positive dataと、自身の顔検出器のtest用のnegative dataとして使用。

UIUC Image Database for Car Detection (http://cogcomp.org/Data/Car/)
->顔検出器の　train用の　negative dataと、自身の顔検出器のtrain用のnegative dataとして使用。

## Reference
HOG特徴量とSVMを使った自動車の検出(http://96n.hatenablog.com/entry/2016/01/23/100311)
Non-Maximum Suppression in Python (https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/)


## Author

[uvazke](https://github.com/uvazke)


