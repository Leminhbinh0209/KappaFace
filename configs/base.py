from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "kappaface"
config.network = "r100"
config.resume = False
config.output = "emore_kappaface_r100_T04_m08_wsamples"
config.instance = "emore_kappaface_r100_T04_m08_wsamples"
config.dataset = "emore"
config.embedding_size = 512
config.sample_rate = 1
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.T = 0.4
config.m = 0.8
config.gamma = 0.7
config.lr = 0.1  # batch size is 512
config.momentum = True  # momentum encoder

if config.dataset == "emore":
    config.rec = "./faces_emore"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch = 24
    config.warmup_epoch = -1
    config.decay_epoch = [10, 18, 22]
    config.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]

elif config.dataset == "ms1m-retinaface-t1":
    config.rec = "./ms1m-retinaface-t1"
    config.num_classes = 93431
    config.num_image = 5179510
    config.num_epoch = 25
    config.warmup_epoch = -1
    config.decay_epoch = [11, 17, 22]
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

elif config.dataset == "glint360k":
    config.rec = "./glint360k"
    config.num_classes = 360232
    config.num_image = 17091657
    config.num_epoch = 20
    config.warmup_epoch = -1
    config.decay_epoch = [8, 12, 15, 18]
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

elif config.dataset == "webface":
    config.rec = "./faces_webface_112x112"
    config.num_classes = 10572
    config.num_image = "forget"
    config.num_epoch = 50
    config.warmup_epoch = -1
    config.decay_epoch = [10, 28, 38, 46]
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
