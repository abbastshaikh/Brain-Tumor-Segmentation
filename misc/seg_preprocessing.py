import numpy as np
import pandas as pd
import nibabel as nib
from keras.utils import to_categorical
from tqdm import tqdm

import warnings
warnings.filterwarnings(action = 'once')

print("Setup Complete")
mount = "MICCAI_BraTS2020_TrainingData"

name_mapping = pd.read_csv(mount + "/name_mapping.csv")

ids = name_mapping['BraTS_2020_subject_ID']
 
for i in tqdm(range(len(ids))):

    labels = [1, 2, 4]
    seg = nib.load(mount + "/{}/{}_seg.nii.gz".format(ids[i], ids[i]))
    data = np.array(seg.get_fdata())

    seg_wt = data.copy()
    seg_wt[seg_wt > 0] = 1

    data = to_categorical(data, 5)[:, :, :, labels]
    
    seg_tc = data[:, :, :, 0]
    seg_ed = data[:, :, :, 1]
    seg_et = data[:, :, :, 2]

    seg_tc = nib.Nifti1Image(seg_tc, affine = seg.affine)
    nib.save(seg_tc, mount + "/{}/{}_seg_tc.nii.gz".format(ids[i], ids[i]))

    seg_ed = nib.Nifti1Image(seg_ed, affine = seg.affine)
    nib.save(seg_ed, mount + "/{}/{}_seg_ed.nii.gz".format(ids[i], ids[i]))

    seg_et = nib.Nifti1Image(seg_et, affine = seg.affine)
    nib.save(seg_et, mount + "/{}/{}_seg_et.nii.gz".format(ids[i], ids[i]))
    seg_wt = nib.Nifti1Image(seg_wt, affine = seg.affine)
    nib.save(seg_wt, mount + "/{}/{}_seg_wt.nii.gz".format(ids[i], ids[i]))