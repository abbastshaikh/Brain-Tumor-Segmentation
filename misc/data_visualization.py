import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action = 'once')

dataDir = "MICCAI_BraTS2020_TrainingData"


#%% Load Data

hgg_id = "BraTS20_Training_001"
lgg_id = "BraTS20_Training_307"

for root, dirs, files in os.walk("{}/{}/".format(dataDir, hgg_id)):
    for file in files:
        
        img = nib.load(root + '/' + file)
        data = img.get_fdata()
        
        if "flair." in file:             
            hgg_flair = np.array(data)        
        if "t1." in file:
            hgg_t1 = np.array(data)
        if "t1ce." in file:
            hgg_t1ce = np.array(data)
        if "t2." in file:
            hgg_t2 = np.array(data)
        if "seg." in file:
            hgg_mask = np.array(data)
            
for root, dirs, files in os.walk("{}/{}/".format(dataDir, lgg_id)):
    for file in files:
        
        img = nib.load(root + '/' + file)
        data = img.get_fdata()
        
        if "flair." in file:             
            lgg_flair = np.array(data)        
        if "t1." in file:
            lgg_t1 = np.array(data)
        if "t1ce." in file:
            lgg_t1ce = np.array(data)
        if "t2." in file:
            lgg_t2 = np.array(data)
        if "seg." in file:
            lgg_mask = np.array(data)
            
#%% Data Visualization

### Tumor Segmentation ###
segmentation = plt.figure()
segmentation.suptitle('Segmentation')
plt.axis('off')
plt.imshow(hgg_flair[:, :, 75], cmap = 'gray')
plt.imshow(hgg_mask[:, :, 75], alpha = 0.5)

leg = segmentation.legend(loc='center', bbox_to_anchor=(0.5, -0.10), shadow=False, ncol=2)

segmentation.tight_layout()
plt.show()
#segmentation.savefig("segmentation.png")


#%%

### Axial, Sagittal, Coronal (HGG) ### 

saggital = np.rot90(hgg_t2[145,])
coronal = np.rot90(hgg_t2[:, 145,])
axial = np.rot90(hgg_t2[:, :, 85])

saggital = np.concatenate([np.zeros([43, 240]), np.concatenate([saggital, np.zeros([42, 240])])])
coronal = np.concatenate([np.zeros([43, 240]), np.concatenate([coronal, np.zeros([42, 240])])])

planes = plt.figure()
planes.suptitle('Saggital, Coronal, and Axial Planes of a T2-weighted MRI', fontsize=14)

planes.add_subplot(1, 3, 1)
plt.axis('off')
plt.imshow(saggital, cmap = 'gray')
plt.title('Saggital', fontsize = 12)

planes.add_subplot(1, 3, 2)
plt.axis('off')
plt.imshow(coronal, cmap = 'gray')
plt.title('Coronal', fontsize = 12)

planes.add_subplot(1, 3, 3)
plt.axis('off')
plt.imshow(axial, cmap = 'gray')
plt.title('Axial', fontsize = 12)

planes.tight_layout()
planes.subplots_adjust(top=1.1)

plt.show()
planes.savefig("planes.png")


### High Grade Glioma ###

indices = [np.count_nonzero(hgg_mask[:, :, i]) for i in range(155)]
max_index = indices.index(max(indices))

hgg = plt.figure(dpi = 300)
hgg.suptitle('High Grade Glioma', fontsize=14)

hgg.add_subplot(2, 5, 1)
plt.axis('off')
plt.imshow(np.rot90(hgg_t1[:, :, max_index]), cmap = 'gray')
plt.title('T1-weighted MRI', fontsize = 6)

hgg.add_subplot(2, 5, 2)
plt.axis('off')
plt.imshow(np.rot90(hgg_t1ce[:, :, max_index]), cmap = 'gray')
plt.title('T1-weighted\nContrast-Enhanced MRI', fontsize = 6)

hgg.add_subplot(2, 5, 3)
plt.axis('off')
plt.imshow(np.rot90(hgg_t2[:, :, max_index]), cmap = 'gray')
plt.title('T2-weighted MRI', fontsize = 6)

hgg.add_subplot(2, 5, 4)
plt.axis('off')
plt.imshow(np.rot90(hgg_flair[:, :, max_index]), cmap = 'gray')
plt.title('T2-weighted\nFLAIR MRI', fontsize = 6)

hgg.add_subplot(2, 5, 5)
plt.axis('off')
plt.imshow(np.rot90(hgg_mask[:, :, max_index]), cmap = 'gray')
plt.title('Segmentation', fontsize = 6)

hgg.add_subplot(2, 5, 6)
plt.axis('off')
plt.imshow(np.rot90(hgg_t1[:, :, max_index]), cmap = 'gray')
plt.imshow(np.rot90(hgg_mask[:, :, max_index]), cmap = 'hot', alpha = 0.3)

hgg.add_subplot(2, 5, 7)
plt.axis('off')
plt.imshow(np.rot90(hgg_t1ce[:, :, max_index]), cmap = 'gray')
plt.imshow(np.rot90(hgg_mask[:, :, max_index]), cmap = 'hot', alpha = 0.3)

hgg.add_subplot(2, 5, 8)
plt.axis('off')
plt.imshow(np.rot90(hgg_t2[:, :, max_index]), cmap = 'gray')
plt.imshow(np.rot90(hgg_mask[:, :, max_index]), cmap = 'hot', alpha = 0.3)
plt.title('Segmentation Transposed on MRIs', fontsize = 6)

hgg.add_subplot(2, 5, 9)
plt.axis('off')
plt.imshow(np.rot90(hgg_flair[:, :, max_index]), cmap = 'gray')
plt.imshow(np.rot90(hgg_mask[:, :, max_index]), cmap = 'hot', alpha = 0.3)

hgg.add_subplot(2, 5, 10)
plt.axis('off')
plt.imshow(np.rot90(hgg_mask[:, :, max_index]), cmap = 'hot')

hgg.tight_layout()
hgg.subplots_adjust(top=0.97)
hgg.subplots_adjust(hspace = -0.3)

plt.show()
hgg.savefig("hgg.png")


### Low Grade Glioma ###

indices = [np.count_nonzero(lgg_mask[:, :, i]) for i in range(155)]
max_index = indices.index(max(indices))

lgg = plt.figure(dpi = 300)
lgg.suptitle('Low Grade Glioma', fontsize=14)

lgg.add_subplot(2, 5, 1)
plt.axis('off')
plt.imshow(np.rot90(lgg_t1[:, :, max_index]), cmap = 'gray')
plt.title('T1-weighted MRI', fontsize = 6)

lgg.add_subplot(2, 5, 2)
plt.axis('off')
plt.imshow(np.rot90(lgg_t1ce[:, :, max_index]), cmap = 'gray')
plt.title('T1-weighted\nContrast-Enhanced MRI', fontsize = 6)

lgg.add_subplot(2, 5, 3)
plt.axis('off')
plt.imshow(np.rot90(lgg_t2[:, :, max_index]), cmap = 'gray')
plt.title('T2-weighted MRI', fontsize = 6)

lgg.add_subplot(2, 5, 4)
plt.axis('off')
plt.imshow(np.rot90(lgg_flair[:, :, max_index]), cmap = 'gray')
plt.title('T2-weighted\nFLAIR MRI', fontsize = 6)

lgg.add_subplot(2, 5, 5)
plt.axis('off')
plt.imshow(np.rot90(lgg_mask[:, :, max_index]), cmap = 'gray')
plt.title('Segmentation', fontsize = 6)

lgg.add_subplot(2, 5, 6)
plt.axis('off')
plt.imshow(np.rot90(lgg_t1[:, :, max_index]), cmap = 'gray')
plt.imshow(np.rot90(lgg_mask[:, :, max_index]), cmap = 'hot', alpha = 0.3)

lgg.add_subplot(2, 5, 7)
plt.axis('off')
plt.imshow(np.rot90(lgg_t1ce[:, :, max_index]), cmap = 'gray')
plt.imshow(np.rot90(lgg_mask[:, :, max_index]), cmap = 'hot', alpha = 0.3)

lgg.add_subplot(2, 5, 8)
plt.axis('off')
plt.imshow(np.rot90(lgg_t2[:, :, max_index]), cmap = 'gray')
plt.imshow(np.rot90(lgg_mask[:, :, max_index]), cmap = 'hot', alpha = 0.3)
plt.title('Segmentation Transposed on MRIs', fontsize = 6)

lgg.add_subplot(2, 5, 9)
plt.axis('off')
plt.imshow(np.rot90(lgg_flair[:, :, max_index]), cmap = 'gray')
plt.imshow(np.rot90(lgg_mask[:, :, max_index]), cmap = 'hot', alpha = 0.3)

lgg.add_subplot(2, 5, 10)
plt.axis('off')
plt.imshow(np.rot90(lgg_mask[:, :, max_index]), cmap = 'hot')

lgg.tight_layout()
lgg.subplots_adjust(top=0.97)
lgg.subplots_adjust(hspace = -0.3)

plt.show()
lgg.savefig("lgg.png")