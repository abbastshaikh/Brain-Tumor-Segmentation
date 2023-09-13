import keras.backend as K
import numpy as np

# Keras Metrics

def dice(y_true, y_pred, weighted = True):

    class_num = 4
   
    for i in range(class_num):
        
        y_true_f = K.flatten(y_true[..., i])
        y_pred_f = K.flatten(y_pred[..., i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon()))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
               
    total_loss = total_loss / class_num
    
    return total_loss

def dice_wt(y_true, y_pred):

    class_weights = [0, 1, 1, 1]
   
    for i in range(len(class_weights)):
        
        y_true_f = K.flatten(y_true[..., i])
        y_pred_f = K.flatten(y_pred[..., i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + K.epsilon() * class_weights[i]) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon()))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
               
    total_loss = total_loss / sum(class_weights)
    
    return total_loss

def dice_tc(y_true, y_pred):

    class_weights = [0, 1, 0, 1]
   
    for i in range(len(class_weights)):
        
        y_true_f = K.flatten(y_true[..., i])
        y_pred_f = K.flatten(y_pred[..., i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + K.epsilon() * class_weights[i]) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon()))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
               
    total_loss = total_loss / sum(class_weights)
    
    return total_loss

def dice_en(y_true, y_pred):

    class_weights = [0, 0, 0, 1]
   
    for i in range(len(class_weights)):
        
        y_true_f = K.flatten(y_true[..., i])
        y_pred_f = K.flatten(y_pred[..., i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + K.epsilon() * class_weights[i]) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon()))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
               
    total_loss = total_loss / sum(class_weights)
    
    return total_loss

def dice_coef_loss(y_true, y_pred):
    return 1. - dice(y_true, y_pred)

def loss_func(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred)


# Numpy Metrics

def binary_dice3d(s,g):
    #dice score of two 3D volumes
    num = np.sum(np.multiply(s, g))
    denom = s.sum() + g.sum() 
    if denom == 0:
        return 1
    else:
        return  2.0*num/denom

def sensitivity (seg,ground): 
    #computs false negative rate
    num=np.sum(np.multiply(ground, seg ))
    denom=np.sum(ground)
    if denom==0:
        return 1
    else:
        return  num/denom

def specificity (seg,ground): 
    #computes false positive rate
    num=np.sum(np.multiply(ground==0, seg ==0))
    denom=np.sum(ground==0)
    if denom==0:
        return 1
    else:
        return  num/denom
    
def DSC_whole(pred, orig_label):
    #computes dice for the whole tumor
    return binary_dice3d(pred>0,orig_label>0)

def DSC_en(pred, orig_label):
    #computes dice for enhancing region
    return binary_dice3d(pred==4,orig_label==4)

def DSC_core(pred, orig_label):
    #computes dice for core region
    seg_=np.copy(pred)
    ground_=np.copy(orig_label)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return binary_dice3d(seg_>0,ground_>0)

def sensitivity_whole (seg,ground):
    return sensitivity(seg>0,ground>0)

def sensitivity_en (seg,ground):
    return sensitivity(seg==4,ground==4)

def sensitivity_core (seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return sensitivity(seg_>0,ground_>0)

def specificity_whole (seg,ground):
    return specificity(seg>0,ground>0)

def specificity_en (seg,ground):
    return specificity(seg==4,ground==4)

def specificity_core (seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return specificity(seg_>0,ground_>0)

def accuracy(predictions, labels):
    return np.sum(np.isclose(labels, predictions)) / 240 / 240 / 155