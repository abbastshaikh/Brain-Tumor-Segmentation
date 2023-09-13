from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, MaxPooling2D, UpSampling2D, Activation, AveragePooling2D, Lambda
from keras.layers import concatenate, add, multiply
from keras.models import Model
import keras.backend as K

initializer = 'he_normal'
dropout_rate = 0.2


def unet(input_size): 
    
    # INPUT #    
    flair_input = Input(shape = input_size, name = 'flair_input')
    t1_input = Input(shape = input_size, name = 't1_input')
    t1ce_input = Input(shape = input_size, name = 't1ce_input')
    t2_input = Input(shape = input_size, name = 't2_input')
    
    # U-NET # 
    all_input = concatenate([flair_input, t1_input, t1ce_input, t2_input])
    
    conv1_1 = Conv2D(48, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(all_input)
    b1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(48, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(b1)
    dropout1 = Dropout(dropout_rate)(conv1_2)
    pool1 = MaxPooling2D()(dropout1)
  
    conv2_1 = Conv2D(96, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool1)
    b2 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(96, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(b2)
    dropout2 = Dropout(dropout_rate)(conv2_2)
    pool2 = MaxPooling2D()(dropout2)
       
    conv3_1 = Conv2D(192, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool2)
    b3 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(192, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(b3)
    dropout3 = Dropout(dropout_rate)(conv3_2)
    pool3 = MaxPooling2D()(dropout3)
   
    conv4_1 = Conv2D(384, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool3)
    conv4_2 = Conv2D(384, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv4_1)
    b4 = BatchNormalization()(conv4_2)
    conv4_3 = Conv2D(384, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(b4)
    dropout4 = Dropout(dropout_rate)(conv4_3)
       
    conv5_1 = Conv2D(192, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2, 2))(dropout4))
    merge5 = concatenate([conv3_2, conv5_1])
    b5 = BatchNormalization()(merge5)
    conv5_2 = Conv2D(192, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(b5)
    dropout5 = Dropout(dropout_rate)(conv5_2)
   
    conv6_1 = Conv2D(96, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2, 2))(dropout5))
    merge6 = concatenate([conv2_2, conv6_1])
    b6 = BatchNormalization()(merge6)
    conv6_2 = Conv2D(96, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(b6)
    dropout6 = Dropout(dropout_rate)(conv6_2)
    
    conv7_1 = Conv2D(48, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2, 2))(dropout6))
    merge7 = concatenate([conv1_2, conv7_1])
    b7 = BatchNormalization()(merge7)
    conv7_2 = Conv2D(48, (3, 3), activation = 'relu', padding = 'same', kernel_initializer =initializer)(b7)
    dropout7 = Dropout(dropout_rate)(conv7_2)

    logits = Conv2D(4, (1, 1), name = 'output')(dropout7) 
    seg_output = Activation('softmax')(logits)
    
    # CREATE MODEL #
    model = Model(inputs = [flair_input, t1_input, t1ce_input, t2_input], outputs = seg_output)
    print(model.summary())

    return model

model = unet((240, 240, 1))


def wasp_net(input_size): 
    
    # INPUT #    
    flair_input = Input(shape = input_size, name = 'flair_input')
    t1_input = Input(shape = input_size, name = 't1_input')
    t1ce_input = Input(shape = input_size, name = 't1ce_input')
    t2_input = Input(shape = input_size, name = 't2_input')
    
    # U-NET # 
    all_input = concatenate([flair_input, t1_input, t1ce_input, t2_input])
    
    # Encoder 
    conv1_1 = Conv2D(48, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(all_input)
    dropout1_1 = Dropout(dropout_rate)(conv1_1)
    conv1_2 = Conv2D(48, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(dropout1_1)
    dropout1_2 = Dropout(dropout_rate)(conv1_2)
    pool1 = MaxPooling2D()(dropout1_2)
  
    conv2_1 = Conv2D(96, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool1)
    dropout2_1 = Dropout(dropout_rate)(conv2_1)
    conv2_2 = Conv2D(96, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(dropout2_1)
    dropout2_2 = Dropout(dropout_rate)(conv2_2)
    pool2 = MaxPooling2D()(dropout2_2)
       
    conv3_1 = Conv2D(192, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool2)
    dropout3_1 = Dropout(dropout_rate)(conv3_1)
    conv3_2 = Conv2D(192, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(dropout3_1)
    dropout3_2 = Dropout(dropout_rate)(conv3_2)
    pool3 = MaxPooling2D()(dropout3_2)
   
    conv4_1 = Conv2D(384, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool3)
    conv4_2 = Conv2D(384, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv4_1)
    dropout4_1 = Dropout(dropout_rate)(conv4_2)
    conv4_3 = Conv2D(384, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(dropout4_1)
    dropout4_2 = Dropout(dropout_rate)(conv4_3)
    
    # WASP
    wasp1 = Conv2D(96, (3, 3), dilation_rate = 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(dropout4_2)    
    wasp2 = Conv2D(96, (3, 3), dilation_rate = 6, activation = 'relu', padding = 'same', kernel_initializer = initializer)(wasp1)
    wasp3 = Conv2D(96, (3, 3), dilation_rate = 9, activation = 'relu', padding = 'same', kernel_initializer = initializer)(wasp2)   
    wasp4 = Conv2D(96, (3, 3), dilation_rate = 12, activation = 'relu', padding = 'same', kernel_initializer = initializer)(wasp3)
    
    wasp_pool = AveragePooling2D(pool_size=(2, 2))(dropout4_2)
    wasp_pool = UpSampling2D((2, 2), interpolation='bilinear')(wasp_pool)
   
    concat = Conv2D(96, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concatenate([wasp1, wasp2, wasp3, wasp4, wasp_pool]))     

    # Decoder
    conv5_1 = Conv2D(192, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2, 2))(concat))
    merge5 = concatenate([conv3_2, conv5_1])
    dropout5_1 = Dropout(dropout_rate)(merge5)
    conv5_2 = Conv2D(192, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(dropout5_1)
    dropout5_2 = Dropout(dropout_rate)(conv5_2)
   
    conv6_1 = Conv2D(96, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2, 2))(dropout5_2))
    merge6 = concatenate([conv2_2, conv6_1])
    dropout6_1 = Dropout(dropout_rate)(merge6)
    conv6_2 = Conv2D(96, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(dropout6_1)
    dropout6_2 = Dropout(dropout_rate)(conv6_2)
    
    conv7_1 = Conv2D(48, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2, 2))(dropout6_2))
    merge7 = concatenate([conv1_2, conv7_1])
    dropout7_1 = Dropout(dropout_rate)(merge7)
    conv7_2 = Conv2D(48, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(dropout7_1)
    
    seg_output = Conv2D(4, 1, padding = 'same', activation = 'softmax')(conv7_2)
    
    # CREATE MODEL #
    model = Model(inputs = [flair_input, t1_input, t1ce_input, t2_input], outputs = seg_output)
    print(model.summary())

    return model


model = wasp_net((240, 240, 1))

def wasp_attention_net(input_size): 
    
    # INPUT #    
    flair_input = Input(shape = input_size, name = 'flair_input')
    t1_input = Input(shape = input_size, name = 't1_input')
    t1ce_input = Input(shape = input_size, name = 't1ce_input')
    t2_input = Input(shape = input_size, name = 't2_input')
    
    # Encoder
    all_input = concatenate([flair_input, t1_input, t1ce_input, t2_input])
    
    conv1_1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(all_input)
    b1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(b1)
    dropout1 = Dropout(0.4)(conv1_2)
    pool1 = MaxPooling2D()(dropout1)
  
    conv2_1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    b2 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(b2)
    dropout2 = Dropout(0.4)(conv2_2)
    pool2 = MaxPooling2D()(dropout2)
       
    conv3_1 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    b3 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(b3)
    dropout3 = Dropout(0.4)(conv3_2)
    pool3 = MaxPooling2D()(dropout3)
    
    # Bottleneck
    conv4_1 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4_2 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_1)
    b4 = BatchNormalization()(conv4_2)
    conv4_3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(b4)
    dropout4 = Dropout(0.4)(conv4_3)
    
    # WASP
    wasp1 = Conv2D(128, (3, 3), dilation_rate = 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_uniform')(dropout4)    
    wasp2 = Conv2D(128, (3, 3), dilation_rate = 6, activation = 'relu', padding = 'same', kernel_initializer = 'random_uniform')(wasp1)
    wasp3 = Conv2D(128, (3, 3), dilation_rate = 9, activation = 'relu', padding = 'same', kernel_initializer = 'random_uniform')(wasp2)   
    wasp4 = Conv2D(128, (3, 3), dilation_rate = 12, activation = 'relu', padding = 'same', kernel_initializer = 'random_uniform')(wasp3)
    
    wasp_pool = AveragePooling2D(pool_size=(2, 2))(dropout4)
    wasp_pool = UpSampling2D((2, 2), interpolation='bilinear')(wasp_pool)
   
    concat = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concatenate([wasp1, wasp2, wasp3, wasp4, wasp_pool]))   
    
    # Decoder
    gating1 = gating_signal(concat, 128, True)
    att1 = attention_block(conv3_2, gating1, 128) 
    
    up5 = UpSampling2D(size = (2, 2))(concat)
    merge5 = concatenate([up5, att1])
    conv5_1 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5_2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_1)
    dropout5 = Dropout(0.4)(conv5_2)
   
    gating2 = gating_signal(concat, 64, True)
    att2 = attention_block(conv2_2, gating2, 64) 
    
    up6 = UpSampling2D(size = (2, 2))(dropout5)
    merge6 = concatenate([up6, att2])
    conv6_1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6_2 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6_1)
    dropout6 = Dropout(0.4)(conv6_2)
    
    gating3 = gating_signal(concat, 32, True)
    att3 = attention_block(conv1_2, gating3, 32) 
    
    up7 = UpSampling2D(size = (2, 2))(dropout6)
    merge7 = concatenate([up7, att3])
    conv7_1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7_2 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7_1)
    dropout7 = Dropout(0.4)(conv7_2)
    
    seg_output = Conv2D(4, 1, padding = 'same', activation = 'softmax')(dropout7)
    
    # CREATE MODEL #
    model = Model(inputs = [flair_input, t1_input, t1ce_input, t2_input], outputs = seg_output)
    print(model.summary())

    return model


def expend_as(tensor, rep):
     return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)

def gating_signal(input, out_size, batch_norm=False):
    
    x = Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3])

    y = multiply([upsample_psi, x])

    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn
