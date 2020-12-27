cfg = {}
cfg['batch_size'] = 64 # batch size of input
cfg['learning_rate'] = 1.2e-3# learning rate to be used
cfg['epochs'] =  100 # number of epochs for which the model is trained
cfg['cuda'] = True #True or False depending whether you want to run your model on a GPU or not. If you set this to True, make sure to start a GPU pod on ieng6 server
cfg['train'] = True# True or False; True denotes that the model is bein deployed in training mode, False means the model is not being used to generate reviews
cfg['prune'] = False
cfg['prune_percent'] = 0