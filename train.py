import argparse
import os
import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

from config import patience, epochs, num_train_samples, num_valid_samples, batch_size
from data_generator import train_gen, valid_gen
import separable_model
import model
from utils import get_available_gpus, categorical_crossentropy_color

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pretrained", help="path to save pretrained model files")
    ap.add_argument("--name-dir", type=str)
    ap.add_argument("--image-dir", type=str)
    ap.add_argument("--gpu", default='0', type=str)
    ap.add_argument("--epoch", default=100, type=int)
    ap.add_argument("--model", default='original', type=str)
    args = vars(ap.parse_args())
    pretrained_path = args["pretrained"]
    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]

    checkpoint_models_path = 'models/%s'%args["model"]
    os.makedirs(checkpoint_models_path, exist_ok=True)
    print(checkpoint_models_path)
    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs/%s'%args["model"], histogram_freq=0, write_graph=True, write_images=True)
    model_names = os.path.join(checkpoint_models_path, 'model.{epoch:02d}-{val_loss:.4f}.hdf5')
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, period=25)
#     early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.02, patience=int(patience / 4), verbose=1)


    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            if epoch % 25 == 0:
                fmt = checkpoint_models_path + 'model.%02d-%.4f.hdf5'
                self.model_to_save.save(fmt % (epoch, logs['val_loss']))


    # Load our model, added support for Multi-GPUs
    num_gpu = len(get_available_gpus())
    if num_gpu >= 2:
        with tf.device("/cpu:0"):
            model = model.build_model()
            if args["model"] != "original":
                model = separable_model.build_model()
            if pretrained_path is not None:
                model.load_weights(pretrained_path)
                
                for layer in new_model.layers[:-4]:
                    layer.trainable = False
        new_model = multi_gpu_model(model, gpus=num_gpu)
        # rewrite the callback: saving through the original model and not the multi-gpu model.
        model_checkpoint = MyCbk(model)
    else:
        new_model = model.build_model()
        if args["model"] != "original":
            new_model = separable_model.build_model()
        if pretrained_path is not None:
            new_model.load_weights(pretrained_path)
            
            for layer in new_model.layers[:-1]:
                layer.trainable = False
            
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True, clipnorm=5.)
    adam = keras.optimizers.Adam(lr=3e-5, beta_1=0.9, beta_2=0.999, decay=0.001)
    new_model.compile(optimizer=adam, loss='categorical_crossentropy')
    
    print(new_model.summary())

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint]
    
    train_set = train_gen(name_dir=args["name_dir"], image_dir=args["image_dir"])
    valid_set = valid_gen(name_dir=args["name_dir"], image_dir=args["image_dir"])
    # Start Fine-tuning
    new_model.fit_generator(train_set,
                            steps_per_epoch=len(train_set),
                            validation_data=valid_set,
                            validation_steps=len(valid_set),
                            epochs=args["epoch"],
                            verbose=1,
                            callbacks=callbacks,
                            use_multiprocessing=True,
                            workers=4
                            )
