import os
import nibabel as nib
import numpy as np
import pandas as pd

from keras.models import load_model
from keras.utils import to_categorical

from metrics import *

import warnings
warnings.filterwarnings(action = 'once')

mount = "/data"
name_mapping = pd.read_csv(mount + "/name_mapping.csv")
ids = name_mapping['BraTS_2020_subject_ID']
ids = ids[300:]

#%% Load Models
unet = load_model(".h5", custom_objects={'loss_func': loss_func, 'dice': dice})
wasp_z = load_model(".h5", custom_objects={'loss_func': loss_func, 'dice': dice})
wasp_y = load_model(".h5", custom_objects={'loss_func': loss_func, 'dice': dice})
wasp_x = load_model(".h5", custom_objects={'loss_func': loss_func, 'dice': dice})

def combine (ax, cor, sag): 
    combined_prob = np.add(np.add(ax, cor), sag)
    combined = np.argmax(combined_prob, axis = -1)
    return combined

#%% Generate Predictions

unet_wt_dice = []
unet_et_dice = []
unet_tc_dice = []
unet_wt_sens = []
unet_wt_spec = []
unet_wt_accuracy = []

prop_wt_dice = []
prop_et_dice = []
prop_tc_dice = []
prop_wt_sens = []
prop_wt_spec = []
prop_wt_accuracy = []

ax_wt_dice = []
ax_et_dice = []
ax_tc_dice = []
ax_wt_sens = []
ax_wt_spec = []
ax_wt_accuracy = []

for brats_id in ids:
    
    print(brats_id)
    
    path = mount + "/" + brats_id + "/"            
         
    for root, dirs, files in os.walk(path):
            
        for file in files:
            
            img = nib.load(root + '/' + file)
            data = np.array(img.get_fdata())
            
            if "seg" not in file:
                data = (data-np.min(data))/(np.max(data)-np.min(data))
        
            data = np.swapaxes(data, 0, 2)
            data = np.swapaxes(data, 1, 2)
            
            if "flair." in file:                        
                flair = np.expand_dims(data, axis = -1)
            
            if "t1." in file:
                t1 = np.expand_dims(data, axis = -1)
            
            if "t1ce." in file:
                t1ce = np.expand_dims(data, axis = -1)
            
            if "t2." in file:
                t2 = np.expand_dims(data, axis = -1)
            
            if "seg." in file: 
                labels = [0, 1, 2, 4]
                data = to_categorical(data, 5)
                masks = np.take(data, labels, axis = -1)
    
    unet_predictions = unet.predict([flair, t1, t1ce, t2], batch_size = 16)
    wasp_z_predictions = wasp_z.predict([flair, t1, t1ce, t2], batch_size = 16)
    unet_predictions = np.argmax(unet_predictions, axis = -1)
    masks_combined = np.argmax(masks, axis = -1)

    for root, dirs, files in os.walk(path):
            
        for file in files:
            
            img = nib.load(root + '/' + file)
            data = np.array(img.get_fdata())
            
            if "seg" not in file:
                data = (data-np.min(data))/(np.max(data)-np.min(data))
                
            data = np.append(data, np.zeros((240, 240, 5)), axis = -1)   
            data = np.swapaxes(data, 0, 1)
            
            if "flair." in file:                        
                flair = np.expand_dims(data, axis = -1)
            
            if "t1." in file:
                t1 = np.expand_dims(data, axis = -1)
            
            if "t1ce." in file:
                t1ce = np.expand_dims(data, axis = -1)
            
            if "t2." in file:
                t2 = np.expand_dims(data, axis = -1)
            
            if "seg." in file: 
                labels = [0, 1, 2, 4]
                masks = to_categorical(data, 5)[:, :, :, labels]
                
    wasp_y_predictions = wasp_y.predict([flair, t1, t1ce, t2], batch_size = 16)
    wasp_y_predictions = np.swapaxes(wasp_y_predictions, 0, 2)
    wasp_y_predictions = np.delete(wasp_y_predictions, [155, 156, 157, 158, 159], axis = 0)

    for root, dirs, files in os.walk(path):
            
        for file in files:
            
            img = nib.load(root + '/' + file)
            data = np.array(img.get_fdata())
        
            if "seg" not in file:
                data = (data-np.min(data))/(np.max(data)-np.min(data))
            
            data = np.append(data, np.zeros((240, 240, 5)), axis = -1)   
        
            if "flair." in file:                        
                flair = np.expand_dims(data, axis = -1)
            
            if "t1." in file:
                t1 = np.expand_dims(data, axis = -1)
            
            if "t1ce." in file:
                t1ce = np.expand_dims(data, axis = -1)
            
            if "t2." in file:
                t2 = np.expand_dims(data, axis = -1)
            
            if "seg." in file: 
                labels = [0, 1, 2, 4]
                masks = to_categorical(data, 5)[:, :, :, labels]
                
    wasp_x_predictions = wasp_x.predict([flair, t1, t1ce, t2], batch_size = 16)
    wasp_x_predictions = np.swapaxes(wasp_x_predictions, 0, 2)
    wasp_x_predictions = np.swapaxes(wasp_x_predictions, 1, 2)
    wasp_x_predictions = np.delete(wasp_x_predictions, [155, 156, 157, 158, 159], axis = 0)

    wasp_predictions = combine(wasp_z_predictions, wasp_y_predictions, wasp_x_predictions)
    wasp_z_predictions = np.argmax(wasp_z_predictions, axis = -1)
    
    unet_wt_dice.append(DSC_whole(unet_predictions, masks_combined))
    unet_et_dice.append(DSC_en(unet_predictions, masks_combined))
    unet_tc_dice.append(DSC_core(unet_predictions, masks_combined))
    unet_wt_sens.append(sensitivity_whole(unet_predictions, masks_combined))
    unet_wt_spec.append(specificity_whole(unet_predictions, masks_combined))
    unet_wt_accuracy.append(accuracy(unet_predictions, masks_combined))
    
    prop_wt_dice.append(DSC_whole(wasp_predictions, masks_combined))
    prop_et_dice.append(DSC_en(wasp_predictions, masks_combined))
    prop_tc_dice.append(DSC_core(wasp_predictions, masks_combined))
    prop_wt_sens.append(sensitivity_whole(wasp_predictions, masks_combined))
    prop_wt_spec.append(specificity_whole(wasp_predictions, masks_combined))
    prop_wt_accuracy.append(accuracy(wasp_predictions, masks_combined))
    
    ax_wt_dice.append(DSC_whole(wasp_z_predictions, masks_combined))
    ax_et_dice.append(DSC_en(wasp_z_predictions, masks_combined))
    ax_tc_dice.append(DSC_core(wasp_z_predictions, masks_combined))
    ax_wt_sens.append(sensitivity_whole(wasp_z_predictions, masks_combined))
    ax_wt_spec.append(specificity_whole(wasp_z_predictions, masks_combined))
    ax_wt_accuracy.append(accuracy(wasp_z_predictions, masks_combined))   
    
#%% Return Average Statistics    
print("Average U-Net Whole Tumor Dice: " + str(sum(unet_wt_dice) / len(ids)))
print("Average Proposed Model Whole Tumor Dice: " + str(sum(prop_wt_dice) / len(ids)))
print("Average Axial Model Whole Tumor Dice: " + str(sum(ax_wt_dice) / len(ids)))

print("Average U-Net Enhancing Tumor Dice: " + str(sum(unet_et_dice) / len(ids)))
print("Average Proposed Model Enhancing Tumor Dice: " + str(sum(prop_et_dice) / len(ids)))
print("Average Axial Model Whole Tumor Dice: " + str(sum(ax_et_dice) / len(ids)))
  
print("Average U-Net Tumor Core Dice: " + str(sum(unet_tc_dice) / len(ids)))
print("Average Proposed Model Tumor Core Dice: " + str(sum(prop_tc_dice) / len(ids)))
print("Average Axial Model Whole Tumor Dice: " + str(sum(ax_tc_dice) / len(ids)))

print("Average U-Net Sensitivity: " + str(sum(unet_wt_sens) / len(ids)))
print("Average Proposed Model Sensitivity: " + str(sum(prop_wt_sens) / len(ids)))
print("Average Axial Model Whole Tumor Dice: " + str(sum(ax_wt_sens) / len(ids)))

print("Average U-Net Specificity: " + str(sum(unet_wt_spec) / len(ids)))
print("Average Proposed Model Specificity: " + str(sum(prop_wt_spec) / len(ids)))
print("Average Axial Model Whole Tumor Dice: " + str(sum(ax_wt_spec) / len(ids)))

print("Average U-Net Accuracy: " + str(sum(unet_wt_accuracy) / len(ids)))
print("Average Proposed Model Accuracy: " + str(sum(prop_wt_accuracy) / len(ids)))
print("Average Axial Model Whole Tumor Dice: " + str(sum(ax_wt_accuracy) / len(ids)))
