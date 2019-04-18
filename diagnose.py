"""
reference: https://www.creatis.insa-lyon.fr/Challenge/acdc/code/metrics_acdc.py
"""

import os
import re
import nibabel as nib
from medpy.metric.binary import hd, dc
import numpy as np

class Diagnose(object):
    def __init__(self,
                 from_path,
                 predict_result_path,
                 save_path):
        self.from_path = from_path
        self.predict_result_path = predict_result_path
        self.save_path = save_path

    def conv_int(self, i):
        return int(i) if i.isdigit() else i

    def natural_order(self, sord):
        if isinstance(sord, tuple):
            sord = sord[0]
        return [self.conv_int(c) for c in re.split(r'(\d+)', sord)]

    def load_nii(self, img_path):
        nimg = nib.load(img_path)
        return nimg.get_data(), nimg.affine, nimg.header

    def metrics(self, img_gt, img_pred, voxel_size):
        if img_gt.ndim != img_pred.ndim:
            raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                             "same dimension, {} against {}".format(img_gt.ndim, img_pred.ndim))
        res = []
        # Loop on each classes of the input images
        for c in [3, 1, 2]:
            # Copy the gt image to not alterate the input
            gt_c_i = np.copy(img_gt)
            gt_c_i[gt_c_i != c] = 0

            # Copy the pred image to not alterate the input
            pred_c_i = np.copy(img_pred)
            pred_c_i[pred_c_i != c] = 0

            # Clip the value to compute the volumes
            gt_c_i = np.clip(gt_c_i, 0, 1)
            pred_c_i = np.clip(pred_c_i, 0, 1)

            # Compute the Dice
            dice = dc(gt_c_i, pred_c_i)

            # Compute volume
            volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
            volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

            res += [dice, volpred, volpred-volgt]
        return res


    def heart_metrics(self, seg_3Dmap, voxel_size, classes=[3, 1, 2]):
        """
        Compute the volumes of each classes
        """
        # Loop on each classes of the input images
        volumes = []
        for c in classes:
            # Copy the gt image to not alterate the input
            seg_3Dmap_copy = np.copy(seg_3Dmap)
            seg_3Dmap_copy[seg_3Dmap_copy != c] = 0

            # Clip the value to compute the volumes
            seg_3Dmap_copy = np.clip(seg_3Dmap_copy, 0, 1)

            # Compute volume
            volume = seg_3Dmap_copy.sum() * np.prod(voxel_size) / 1000.
            volumes += [volume]
        return volumes

    def ejection_fraction(self, ed_vol, es_vol):
        """
        Calculate ejection fraction
        """
        stroke_vol = ed_vol - es_vol
        return (np.float(stroke_vol)/np.float(ed_vol))*100


    ### reference from: http://doctor-network.com/Public/LittleTools/325.html
    def diastolic_left_ventricular_volume(self, ed_lv, height, weight):
        body_surface_area_of_man = 0.00607*height + 0.0127*weight - 0.0698
        return ed_lv/(body_surface_area_of_man)


    def diagnose_result(self):
        rows = []
        cases = sorted(next(os.walk(self.from_path))[1])
        for i, case in enumerate(cases):
            current_path = os.path.join(self.from_path, case + '/Info.cfg')
            patient_info = {}
            with open(current_path) as f_in:
                for line in f_in:
                    info = line.rstrip().split(": ")
                    patient_info[info[0]] = info[1]
            current_path = os.path.join(self.from_path, case)
            img_4D, _, hdr = self.load_nii(os.path.join(current_path, "patient%03d_4d.nii.gz" % (i+101)))
            ed_gt, _, _ = self.load_nii(os.path.join(self.predict_result_path, "patient%03d_ED.nii.gz" % (i+101)))
            es_gt, _, _ = self.load_nii(os.path.join(self.predict_result_path, "patient%03d_ES.nii.gz" % (i+101)))
            ed_lv, ed_rv, ed_myo = self.heart_metrics(ed_gt, hdr.get_zooms())
            es_lv, es_rv, es_myo = self.heart_metrics(es_gt, hdr.get_zooms())
            ef_lv = self.ejection_fraction(ed_lv, es_lv)
            ef_rv = self.ejection_fraction(ed_rv, es_rv)
            LVEDVI = self.diastolic_left_ventricular_volume(ed_lv, float(patient_info['Height']), float(patient_info['Weight']))
            myo_ratio = (ed_lv + ed_rv) / ed_myo

            diagnose = 'NOR'
            if ef_lv < 40.0 or LVEDVI > 110.0:
                if LVEDVI > 110.0:
                    diagnose = 'DCM'
                else:
                    diagnose = 'MINF'
            else:
                if myo_ratio < 2.40 and es_myo+ed_myo > 230.0:
                    diagnose = 'HCM'
                elif es_rv > 110.0 or ef_rv < 39.0:
                    diagnose = 'RV'

            print(case, diagnose)
            rows.append([case, patient_info['Height'], patient_info['Weight'],
                         '%.2f' % ed_lv, '%.2f' % es_lv, '%.2f' % ed_rv, '%.2f' % es_rv,
                         '%.2f' % ed_myo, '%.2f' % es_myo, '%.2f' % ef_lv, '%.2f' % ef_rv,
                         '%.2f' % LVEDVI, '%.2f' % myo_ratio, diagnose])

        f = open(self.save_path, 'w')
        for row in rows:
            f.writelines(row[0] + ' ' + row[len(row)-1] + '\n')
        f.close()


if __name__ == "__main__":

    from_path = '../data/testing'
    predict_result_path = '../data/result/predict_nii_gz_result'
    save_path = './diagnose_result.txt'
    dignose = Diagnose(from_path, predict_result_path, save_path)
    dignose.diagnose_result()
