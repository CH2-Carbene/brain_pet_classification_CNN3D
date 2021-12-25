import time,os
def save_model(model,filename,target_column,**kwargs):
    save_model_dir=os.path.join("models",f"{filename}_{target_column}")
    save_model_name="m{}.h5".format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    model.save(os.path.join(save_model_dir,save_model_name))
    model.save("./models/latest.h5")
