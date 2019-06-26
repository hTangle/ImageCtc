from ImageCtc.util.image_generate import gen
from ImageCtc.util.model_generate import captcha_model
from keras.callbacks import ModelCheckpoint

model = captcha_model()
# plot(model,to_file='model.png',show_shapes=True)
checkpointer =ModelCheckpoint(filepath="net-epoch.hdf5", verbose=1, save_best_only=True)
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit_generator(gen(),samples_per_epoch=51200,nb_epoch=5,validation_data=gen(),nb_val_samples=1280,callbacks=[checkpointer])