from unet_clstm_model import *
from image_data_generator import *
from data_preprocess import *
from roi import *
from save_predict_nii_gz import *
from diagnose import *
import cv2
import os
import numpy as np
from keras.callbacks import TensorBoard
from time import time
from keras.models import load_model

def train(data_preprocess):
    unet_clstm = Unet_clstm_model(input_shape=(data_preprocess.result_z, data_preprocess.roi_x, data_preprocess.roi_y, 1))
    model = unet_clstm.res_unet_clstm()
    train_image = data_preprocess.get_roi_image(from_path='../data/training', center_point_csv_path='./train_center_radii.csv')
    train_image = train_image[::, ::, ::, ::, np.newaxis]
    train_label = data_preprocess.get_roi_label(from_path='../data/training', center_point_csv_path='./train_center_radii.csv')

    batch_size = 1
    params = {
          'dim': (data_preprocess.result_z, data_preprocess.roi_x, data_preprocess.roi_y),
          'batch_size': batch_size,
          'n_classes': 4,
          'n_channels': 1,
          'shuffle': False}

    training_generator = DataGenerator(train_image, train_label, **params)
    tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

    model.fit_generator(generator=training_generator, steps_per_epoch=len(train_image)/batch_size, epochs=160, callbacks=[tensorboard])
    model.save('unet_clstm.h5')

def test(data_preprocess, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    model = load_model('unet_clstm.h5')
    test_image = data_preprocess.get_roi_image(from_path='../data/testing', center_point_csv_path='./test_center_radii.csv')
    test_image = test_image[::, ::, ::, ::, np.newaxis]
    for patient in range(test_image.shape[0]):
        predict_test = test_image[patient:patient+1, ::, ::, ::, ::]
        new_pos = model.predict([predict_test, predict_test])
        for count in range(new_pos.shape[1]):
            img = np.zeros((data_preprocess.roi_x, data_preprocess.roi_y))
            img = img.astype('int')
            for current_value in range(4):
                for row in range(data_preprocess.roi_x):
                    for col in range(data_preprocess.roi_y):
                        if new_pos[0, count, row, col, current_value] == max(new_pos[0, count, row, col]):
                            img[row, col] = current_value
            save_name = ''
            if patient % 2 == 0:
                save_name = os.path.join(save_path + str(int(patient/2+1)).zfill(3) + '_01_' + str(count+1).zfill(2) + '_ED.png')
            else:
                save_name = os.path.join(save_path + str(int(patient/2+1)).zfill(3) + '_09_' + str(count+1).zfill(2) + '_ES.png')
            print(patient, count, np.sum(img == 0), np.sum(img == 1), np.sum(img == 2), np.sum(img == 3))
            cv2.imwrite(save_name, img*85)


if __name__ == '__main__':
    # ### Region of interest extraction
    roi_train = ROI('../data/training', './train_center_radii.csv', 1, 101)
    roi_train.save_csv()
    roi_test = ROI('../data/testing', './test_center_radii.csv', 101, 151)
    roi_test.save_csv()

    ### train and test
    data_preprocess = DataPreprocess(roi_x=128, roi_y=128, result_z=21)
    train(data_preprocess)
    test(data_preprocess, '../data/result/test_label_roi_res_unet_bi_clstm_predict_interpolation/')

    ### segmentation
    from_path = '../data/testing'
    predict_png_result_path = '../data/result/test_label_roi_res_unet_bi_clstm_predict_interpolation'
    predict_nii_gz_result_path = './predict_nii_gz_result'
    if not os.path.isdir(predict_nii_gz_result_path):
        os.makedirs(predict_nii_gz_result_path)
    save_data = Save_nii_gz(from_path, predict_png_result_path, predict_nii_gz_result_path)
    save_data.save_data_dir()

    ### dignose
    from_path = '../data/testing'
    predict_result_path = './predict_nii_gz_result'
    save_path = './diagnose_result.txt'
    dignose = Diagnose(from_path, predict_result_path, save_path)
    dignose.diagnose_result()
