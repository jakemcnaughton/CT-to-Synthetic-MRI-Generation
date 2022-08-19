from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = unet()
#Save every epoch
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit(data, target,...)
model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
#model.predict()
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/test",results)
