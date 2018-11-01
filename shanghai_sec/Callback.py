from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


class Callback(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.log_dir = "logs/"
        self.model_dir = "models/"

    def get_tensorboard(self):#是一个可视化的展示器
        tensorboard_callback = TensorBoard(log_dir=self.log_dir, write_grads=True,
                                           histogram_freq=0, write_images=True)
        return tensorboard_callback

    def get_early_stop(self, patience):#当监测值不再改善时，该回调函数将中止训练
        early_stop = EarlyStopping('val_acc', patience=patience)
        return early_stop

    def get_readuce_lr(self, factor, patience):#学习率衰减
        return ReduceLROnPlateau(monitor='val_acc', factor=factor, patience=patience)

    def get_model_ckpt(self):#在每个epoch后保存模型
        model_names = self.model_dir + '.{epoch:02d}-{val_acc:.4f}.h5'
        model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
        return model_checkpoint