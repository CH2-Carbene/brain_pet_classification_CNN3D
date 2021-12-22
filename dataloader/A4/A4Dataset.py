from tensorflow.keras.utils import Sequence 
import nibabel as nib
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import scipy

A4DIR="../../datasets/A4/"

def ndresize(x:np.ndarray,tg_shape):
    orisize=np.array(x.shape)
    tgsize=np.array(tg_shape)
    return scipy.ndimage.zoom(x,tgsize/orisize,order=1)

def random_padcut3d(x:np.ndarray,rate,random_seed=None):
    # dim=len(x.shape)
    if random_seed is not None:
        ts=np.random.get_state()
        np.random.seed(random_seed)

    pc=(x.shape*np.random.uniform(-rate/2.,rate/2.,size=(3))).astype(np.int32)

    pad=np.maximum(0,pc)
    # pad=np.max(np.array((0),dtype=np.int32),pc)
    cut=np.minimum(0,pc)
    cut=tuple([slice(None,None) if c==0 else slice(-c,c) for c in cut])
    
    x=np.pad(x,((pad[0],pad[0]),(pad[1],pad[1]),(pad[2],pad[2])))
    x=x[cut]

    if random_seed is not None:np.random.set_state(ts)
    # len(x.shape)
    # dtlist=[np.pad]
    return x

def loadA4img(bid,ag=False,random_seed=None):
    '''load one A4 img, and can random argument.
    '''
    normalize=lambda x:(x-x.mean())/x.std()
    dirname=os.path.join(A4DIR,"A4_aligned/{}/Florbetapir/").format(bid)
    filename=sorted(os.listdir(dirname))[0]
    fullpath=os.path.join(dirname,filename)
    ngz=nib.load(fullpath)
    x=normalize(ngz.get_fdata())
    if ag:x=random_padcut3d(x,0.15,random_seed)
    x=ndresize(x,(42,50,42))
    return x
    # np.random.get_random()
    

class A4Dataset_train(Sequence):
    '''A4 DataSet batch sequence. Batch_size=0 means only one batch. ag_func must be setted if ag_rate!=0.
    '''
    def __init__(self,x,y,batch_size=0,pre_fx=loadA4img,pre_fy=lambda y:1 if y=="positive" else 0,ag_rate=0): 
        # if ag_rate!=0 and ag_func is None:
        #     raise Exception("Ag_func must be setted when ag_rate is not 0!")
        totlen=len(x)*(ag_rate+1)
        if batch_size==0:batch_size=totlen
        self.x,self.y,self.batch_size=x,y,batch_size
        self.pre_fx,self.pre_fy=pre_fx,pre_fy
        self.ag_rate=ag_rate
        
        
        self.totlen,self.len=totlen,totlen//batch_size+int(totlen%batch_size!=0)
        self.ramp=np.random.permutation(totlen)


        self.ramseedfunc=lambda x,y:(x*114514+y*1919810)%19260817

    def __len__(self):
        return self.len

    def __getitem__(self, b_idx):
        batch_size=self.batch_size
        pre_fx,pre_fy=self.pre_fx,self.pre_fy

        rf=self.ramseedfunc
        la=self.ag_rate+1
        ramp=self.ramp

        batch_x,batch_y=[],[]
        # lst_tid=None
        for i in range(b_idx*batch_size,min(self.totlen,(b_idx+1)*batch_size)):
            tid,agid=ramp[i]//la,ramp[i]%la

            xi,yi=self.x[tid],self.y[tid]
            if agid!=0:
                batch_x.append(pre_fx(xi,ag=True,random_seed=rf(tid,agid)))
            else:
                batch_x.append(pre_fx(xi,ag=False))
            batch_y.append(pre_fy(yi))
            # pre_fx,pre_fy
            # lst_tid=tid

        return np.array(batch_x),np.array(batch_y)
        # batch_x = self.x[idx * self.batch_size:(idx + 1) *
        #     self.batch_size]
        # batch_y = self.y[idx * self.batch_size:(idx + 1) *
        #     self.batch_size]
        # return np.array(list(map(self.pre_fx,batch_x)),np.array(list(map(self.pre_fy,batch_y))))
        # # return super().__getitem__(index)
        
class A4Dataset_test(Sequence):
    '''A4 DataSet batch sequence. Batch_size=0 means only one batch.
    '''
    def __init__(self,x,y,batch_size=0,pre_fx=loadA4img,pre_fy=lambda y:1 if y=="positive" else 0): 
        # if ag_rate!=0 and ag_func is None:
        #     raise Exception("Ag_func must be setted when ag_rate is not 0!")
        totlen=len(x)
        if batch_size==0:batch_size=totlen
        self.x,self.y,self.batch_size=x,y,batch_size
        self.pre_fx,self.pre_fy=pre_fx,pre_fy
        # self.ag_rate=ag_rate
        

        self.totlen,self.len=totlen,totlen//batch_size+int(totlen%batch_size!=0)
        # self.ramp=np.random.permutation(totlen)


        # self.ramseedfunc=lambda x,y:(x*114514+y*1919810)%19260817

    def __len__(self):
        return self.len

    def __getitem__(self, b_idx):
        batch_size=self.batch_size
        pre_fx,pre_fy=self.pre_fx,self.pre_fy

        batch_x,batch_y=[],[]
        # lst_tid=None
        for i in range(b_idx*batch_size,min(self.totlen,(b_idx+1)*batch_size)):
            xi,yi=self.x[i],self.y[i]
            batch_x.append(pre_fx(xi,ag=False))
            batch_y.append(pre_fy(yi))

        return np.array(batch_x),np.array(batch_y)
        

def load_data(csv_name,target_column,max_size=-1):
    '''load data from document/fileName.csv. y is target_column
    '''
    c_bid,c_y="BID",target_column
    unusableBID_set=set(pd.read_csv("./unusable.csv"))
    df=pd.read_csv(os.path.join(A4DIR,"./document",csv_name))[[c_bid,c_y]]
    df=df.drop_duplicates()
    if True in set(df.duplicated([c_bid])):
        raise Exception("Different {} with same BID!".format(c_y))

    x,y=[],[]
    unusable=[]
    with tqdm(total=len(df)) as pbar:

        for line in df.iloc():
            if len(x)==max_size:
                break
            try:
                bid,ly=line[c_bid],line[c_y]
                if bid in unusableBID_set:raise Exception("BID_{} in unusable list".format(bid))

                dirname=os.path.join(A4DIR,"A4_aligned/{}/Florbetapir/").format(bid)
                filename=sorted(os.listdir(dirname))[0]
                fullpath=os.path.join(dirname,filename)
                
                # if os.path.getsize(fullpath)>=100*1024 and os.path.getsize(fullpath)<=200*1024:
                #     print(fullpath)
                if os.path.getsize(fullpath)<=100*1024:
                    raise Exception("BID_{} image data too small".format(bid))
                ngz=nib.load(fullpath)
                x.append(bid)
                y.append(ly)

            except Exception as e:
                # print(e)
                unusable.append(bid)
                # break

            pbar.update(1)

    # x,y=list(df[c_bid]),list(df[c_y])
    # self.batch_size=batch_size
    print("Total data number:{}".format(len(x)))
    print("unusable BID:{}".format(unusable))
    return x,y


def test1():
    x,y=load_data("A4_PETVADATA_PRV2.csv","SCORE")
    print(len(x),len(y))
def test_2():
    print(random_padcut3d(np.random.random((100,100,100)),rate=0.15,random_seed=114514).shape)
    print(loadA4img("B10018169",False).shape)
def total_test():
    x,y=load_data("A4_PETVADATA_PRV2.csv","SCORE",100)
    train_d=A4Dataset_train(x,y,64,ag_rate=4)
    print(train_d.len)
    for i,j in train_d:print(i.shape,np.max(i),j.shape)

    test_d=A4Dataset_test(x,y)
    print(test_d.len)
    for i,j in test_d:print(i.shape,np.max(i),j.shape)

if __name__ == '__main__':
   total_test()
