import numpy as np
import pandas as pd
import radiomics
import six
from tqdm import tqdm

import warnings
warnings.filterwarnings(action = 'once')

mount = "MICCAI_BraTS2020_TrainingData"
print("Setup Complete")

name_mapping = pd.read_csv(mount + "/name_mapping.csv")

ids_df = pd.DataFrame({'BraTS_2020_subject_ID' : name_mapping['BraTS_2020_subject_ID']})
ids = list(ids_df['BraTS_2020_subject_ID'])

shape_features_names = ['com_x', 
                        'com_y', 
                        'com_z', 
                        'original_shape_Elongation', 
                        'original_shape_Flatness', 
                        'original_shape_LeastAxisLength', 
                        'original_shape_MajorAxisLength', 
                        'original_shape_Maximum2DDiameterColumn',
                        'original_shape_Maximum2DDiameterRow',
                        'original_shape_Maximum2DDiameterSlice', 
                        'original_shape_Maximum3DDiameter', 
                        'original_shape_MeshVolume', 
                        'original_shape_MinorAxisLength', 
                        'original_shape_Sphericity', 
                        'original_shape_SurfaceArea', 
                        'original_shape_SurfaceVolumeRatio',
                        'original_shape_VoxelVolume']

empty_shape_dict = {shape_features_names[i] : 0 for i in range(len(shape_features_names))}

other_feature_names = ['original_firstorder_10Percentile', 
                       'original_firstorder_90Percentile',
                       'original_firstorder_Energy',
                       'original_firstorder_Entropy', 
                       'original_firstorder_InterquartileRange',
                       'original_firstorder_Kurtosis', 
                       'original_firstorder_Maximum',
                       'original_firstorder_MeanAbsoluteDeviation', 
                       'original_firstorder_Mean', 
                       'original_firstorder_Median', 
                       'original_firstorder_Minimum',  
                       'original_firstorder_Range', 
                       'original_firstorder_RobustMeanAbsoluteDeviation',  
                       'original_firstorder_RootMeanSquared', 
                       'original_firstorder_Skewness', 
                       'original_firstorder_TotalEnergy', 
                       'original_firstorder_Uniformity', 
                       'original_firstorder_Variance',
                       
                       'original_glcm_Autocorrelation', 
                       'original_glcm_ClusterProminence', 
                       'original_glcm_ClusterShade',
                       'original_glcm_ClusterTendency', 
                       'original_glcm_Contrast',
                       'original_glcm_Correlation',
                       'original_glcm_DifferenceAverage',
                       'original_glcm_DifferenceEntropy',
                       'original_glcm_DifferenceVariance',
                       'original_glcm_Id',
                       'original_glcm_Idm', 
                       'original_glcm_Idmn', 
                       'original_glcm_Idn',  
                       'original_glcm_Imc1', 
                       'original_glcm_Imc2', 
                       'original_glcm_InverseVariance', 
                       'original_glcm_JointAverage', 
                       'original_glcm_JointEnergy', 
                       'original_glcm_JointEntropy', 
                       'original_glcm_MCC', 
                       'original_glcm_MaximumProbability', 
                       'original_glcm_SumAverage',
                       'original_glcm_SumEntropy', 
                       'original_glcm_SumSquares', 
                           
                       'original_gldm_DependenceEntropy',
                       'original_gldm_DependenceNonUniformity', 
                       'original_gldm_DependenceNonUniformityNormalized', 
                       'original_gldm_DependenceVariance',  
                       'original_gldm_GrayLevelNonUniformity', 
                       'original_gldm_GrayLevelVariance', 
                       'original_gldm_HighGrayLevelEmphasis', 
                       'original_gldm_LargeDependenceEmphasis', 
                       'original_gldm_LargeDependenceHighGrayLevelEmphasis', 
                       'original_gldm_LargeDependenceLowGrayLevelEmphasis', 
                       'original_gldm_LowGrayLevelEmphasis', 
                       'original_gldm_SmallDependenceEmphasis',
                       'original_gldm_SmallDependenceHighGrayLevelEmphasis', 
                       'original_gldm_SmallDependenceLowGrayLevelEmphasis', 
                       
                       'original_glrlm_GrayLevelNonUniformity', 
                       'original_glrlm_GrayLevelNonUniformityNormalized', 
                       'original_glrlm_GrayLevelVariance', 
                       'original_glrlm_HighGrayLevelRunEmphasis', 
                       'original_glrlm_LongRunEmphasis',
                       'original_glrlm_LongRunHighGrayLevelEmphasis',
                       'original_glrlm_LongRunLowGrayLevelEmphasis',
                       'original_glrlm_LowGrayLevelRunEmphasis',
                       'original_glrlm_RunEntropy', 
                       'original_glrlm_RunLengthNonUniformity', 
                       'original_glrlm_RunLengthNonUniformityNormalized', 
                       'original_glrlm_RunPercentage',  
                       'original_glrlm_RunVariance',
                       'original_glrlm_ShortRunEmphasis', 
                       'original_glrlm_ShortRunHighGrayLevelEmphasis', 
                       'original_glrlm_ShortRunLowGrayLevelEmphasis',
                       
                       'original_glszm_GrayLevelNonUniformity', 
                       'original_glszm_GrayLevelNonUniformityNormalized', 
                       'original_glszm_GrayLevelVariance',
                       'original_glszm_HighGrayLevelZoneEmphasis', 
                       'original_glszm_LargeAreaEmphasis', 
                       'original_glszm_LargeAreaHighGrayLevelEmphasis',
                       'original_glszm_LargeAreaLowGrayLevelEmphasis',
                       'original_glszm_LowGrayLevelZoneEmphasis', 
                       'original_glszm_SizeZoneNonUniformity', 
                       'original_glszm_SizeZoneNonUniformityNormalized', 
                       'original_glszm_SmallAreaEmphasis',
                       'original_glszm_SmallAreaHighGrayLevelEmphasis', 
                       'original_glszm_SmallAreaLowGrayLevelEmphasis', 
                       'original_glszm_ZoneEntropy', 
                       'original_glszm_ZonePercentage', 
                       'original_glszm_ZoneVariance', 
                       
                       'original_ngtdm_Busyness',
                       'original_ngtdm_Coarseness', 
                       'original_ngtdm_Complexity', 
                       'original_ngtdm_Contrast', 
                       'original_ngtdm_Strength']
    
wt_tumor_features = pd.DataFrame(columns = shape_features_names)
tc_tumor_features = pd.DataFrame(columns = shape_features_names)
ed_tumor_features = pd.DataFrame(columns = shape_features_names)
et_tumor_features = pd.DataFrame(columns = shape_features_names)

flair_features = pd.DataFrame(columns = other_feature_names)
t1_features = pd.DataFrame(columns = other_feature_names)
t1ce_features = pd.DataFrame(columns = other_feature_names)
t2_features = pd.DataFrame(columns = other_feature_names)

def shape_dict_to_df (results):
   
    features = []
    
    for key, val in six.iteritems(results):
        
        if key.startswith('original_shape_'):
            
            if isinstance(val, np.ndarray):
                features.append(val.item())
            else:
                features.append(val)
                
        if key.startswith('diagnostics_Mask-original_CenterOfMassIndex'):
            
            features.append(val[0])
            features.append(val[1])
            features.append(val[2])
            
        if key.startswith('com'):
            
            features.append(val)
            
    return pd.Series(features, index = wt_tumor_features.columns)

def features_dict_to_df (results):
    
    features = []
    
    for key, val in six.iteritems(results):
        
        if key.startswith('original_'):
            
            if isinstance(val, np.ndarray):
                features.append(val.item())
            else:
                features.append(val)
            
    return pd.Series(features, index = flair_features.columns)


for i in tqdm(range(len(ids))):
      
    flair = mount + "/{}/{}_flair.nii.gz".format(ids[i], ids[i])
    t1 = mount + "/{}/{}_t1.nii.gz".format(ids[i], ids[i])
    t1ce = mount + "/{}/{}_t1ce.nii.gz".format(ids[i], ids[i])
    t2 = mount + "/{}/{}_t2.nii.gz".format(ids[i], ids[i])
     
    seg = mount + "/{}/{}_seg.nii.gz".format(ids[i], ids[i])
    seg_wt = mount + "/{}/{}_seg_wt.nii.gz".format(ids[i], ids[i])
    seg_tc = mount + "/{}/{}_seg_tc.nii.gz".format(ids[i], ids[i])
    seg_ed = mount + "/{}/{}_seg_ed.nii.gz".format(ids[i], ids[i])
    seg_et = mount + "/{}/{}_seg_et.nii.gz".format(ids[i], ids[i])
    
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
      
    
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('shape') 
    
    wt_results = extractor.execute(flair, seg_wt)
    
    while True:
        try:
            tc_results = extractor.execute(flair, seg_tc)
            break
        except ValueError:
            tc_results = empty_shape_dict
            break
        
    while True:
        try:
            ed_results = extractor.execute(flair, seg_ed)
            break
        except ValueError:
            ed_results = empty_shape_dict
            break
        
    while True:
        try:
            et_results = extractor.execute(flair, seg_et)
            break
        except ValueError:
            et_results = empty_shape_dict
            break
     
    wt_tumor_features = wt_tumor_features.append(shape_dict_to_df(wt_results), ignore_index=True)
    tc_tumor_features = tc_tumor_features.append(shape_dict_to_df(tc_results), ignore_index=True)
    ed_tumor_features = ed_tumor_features.append(shape_dict_to_df(ed_results), ignore_index=True)
    et_tumor_features = et_tumor_features.append(shape_dict_to_df(et_results), ignore_index=True)
    
    
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('ngtdm')
    
    
    flair_results = extractor.execute(flair, seg_wt)    
    t1_results = extractor.execute(t1, seg_wt)
    t1ce_results = extractor.execute(t1ce, seg_wt)    
    t2_results = extractor.execute(t2, seg_wt)

    flair_features = flair_features.append(features_dict_to_df(flair_results), ignore_index=True)
    t1_features = t1_features.append(features_dict_to_df(t1_results), ignore_index=True)
    t1ce_features = t1ce_features.append(features_dict_to_df(t1ce_results), ignore_index=True)
    t2_features = t2_features.append(features_dict_to_df(t2_results), ignore_index=True)

    print()

wt_tumor_features.columns = ['wt_' + str(col) for col in wt_tumor_features.columns]
tc_tumor_features.columns = ['tc_' + str(col) for col in tc_tumor_features.columns]
ed_tumor_features.columns = ['ed_' + str(col) for col in ed_tumor_features.columns]
et_tumor_features.columns = ['et_' + str(col) for col in et_tumor_features.columns]
flair_features.columns = ['flair_wt_' + str(col) for col in flair_features.columns]
t1_features.columns = ['t1_wt_' + str(col) for col in t1_features.columns]
t1ce_features.columns = ['t1ce_wt_' + str(col) for col in t1ce_features.columns]
t2_features.columns = ['t2_wt_' + str(col) for col in t2_features.columns]

radiomics = pd.concat([ids_df, 
               wt_tumor_features, 
               tc_tumor_features, 
               ed_tumor_features, 
               et_tumor_features, 
               flair_features, 
               t1_features, 
               t1ce_features, 
               t2_features], axis=1)

radiomics.to_csv(mount + "/radiomics.csv")