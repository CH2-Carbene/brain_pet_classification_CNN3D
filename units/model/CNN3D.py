import tensorflow as tf
from tensorflow.keras import Input,Model,Sequential
from tensorflow.keras import layers,optimizers,losses,metrics
from tensorflow.python.keras.engine import input_layer
# model=keras.Sequential()
# model.add(layers.InputLayer(input_shape=(50,42,42,1)))


#conv3d
def Conv3D_U(channel):
    return layers.Conv3D(channel,3,padding='same')

#BN+activate(relu)
def BN_AC():
    return Sequential([
        layers.BatchNormalization(),
        layers.Activation("relu"),
    ])

#conv3d+BatchNormalization
def Conv3D_BN(channel,dp_rate=0):
    return Sequential([
        Conv3D_U(channel),
        BN_AC(),
        layers.Dropout(dp_rate),
    ])

#Conv3D_Pooling1
def Conv3D_P1(channel):
    return layers.Conv3D(channel,3,strides=(2,2,2))
#Conv3D_Pooling2
def Conv3D_P2(channel):
    return layers.Conv3D(channel,3,strides=(2,2,2),padding='same')
#Conv3D_Pooling+BatchNormalization
# why dropout before BatchNormalization?
def Conv3D_PBN(channel,dp_rate=0):
    return Sequential([
        Conv3D_P2(channel),
        layers.Dropout(dp_rate),
        BN_AC()
    ])

def Merge():
    return layers.Concatenate()
def Liner(units,activation=None):
    return layers.Dense(units,activation=activation)

def CNN3D(cls_num=2):
    ''' Return a 3D-CNN model for classification/regression.
    input shape:(42,50,42,1)
    output shape:(cls_num)
    Args:
      cls_num: int, should be >=1. When cls_num is 1, it's a regression model.
    '''
    is_reg=True if cls_num==1 else False
    L0=Sequential([
        Conv3D_BN(15),
        Conv3D_P1(15)
    ],name="Block0")

    L1=Sequential([
        Conv3D_BN(15,0.2),
        Conv3D_U(15)
    ],name="Block1")
    M1=Merge()

    L2=Sequential([
        BN_AC(),
        Conv3D_PBN(25,0.2),
        Conv3D_U(25)
    ],name="Block2")
    R2=Conv3D_P2(15)
    M2=Merge()

    L3=Sequential([
        BN_AC(),
        Conv3D_PBN(35,0.2),
        Conv3D_U(35)
    ],name="Block3")
    R3=Conv3D_P2(25)
    M3=Merge()
    
    L4=Sequential([
        BN_AC(),
        layers.Conv3D(30,3,padding='valid'),
        layers.Conv3D(30,3,padding='valid')
    ],name="Block4")

    FC=Sequential([
        layers.Flatten(),
        Liner(300,'relu'),
        layers.Dropout(0.2),
        Liner(50,'relu'),
        Liner(cls_num)
    ],name="FC")
    CLF=layers.Softmax()

### network constructure
    inputs = Input(shape=(42,50,42,1), dtype='float32')

    l0_out=L0(inputs)

    l1_x=L1(l0_out)
    l1_y=l0_out
    l1_out=M1([l1_x,l1_y])

    l2_x=L2(l1_out)
    l2_y=R2(l1_out)
    l2_out=M2([l2_x,l2_y])

    l3_x=L3(l2_out)
    l3_y=R3(l2_out)
    l3_out=M3([l3_x,l3_y])

    l4_x=L4(l3_out)

    lfc=FC(l4_x)

    outputs=lfc if is_reg else CLF(lfc)

    opt=optimizers.Adadelta()
    
    loss_func=losses.mse if is_reg else losses.SparseCategoricalCrossentropy()
    metric=[metrics.RootMeanSquaredError() if is_reg else metrics.SparseCategoricalAccuracy()]

    model=Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=opt,loss=loss_func,metrics=metric)
    return model
if __name__=="__main__":
    from tensorflow.keras.utils import plot_model
    model=CNN3D()
    model.summary(line_length=120)
    plot_model(model, "CNN3D.png", show_shapes=True)
