
class parameter():
    def __init__(self):
        self.flags = None
        self.unparsed = None
        self.device = None
        # Train parameters
        self.use_cuda = None
        self.eps = 1e-5
        self.keep_backup = 5
        self.save_interval = 10  # epoches
        self.test_interval = 10  # epoches
        self.dot_interval = 70  # batches

        # Test parameters
        self.evaluate = False
        self.conf_thresh = 0.25
        self.nms_thresh = 0.4
        self.iou_thresh = 0.5

        # no test evalulation
        self.no_eval = False

        self.cfgfile = None
        self.weightfile = None

        self.trainlist      = None
        self.testlist       = None
        self.backupdir      = None
        self.gpus           = None
        self.ngpus          = None
        self.num_workers    = None

        self.batch_size     = None
        self.max_batches    = None
        self.learning_rate  = None
        self.momentum       = None
        self.decay          = None
        self.steps          = None
        self.scales         = None

        self.max_epochs     = None
        self.init_epoch     = None
        self.loss_layers    = None

        self.optimizer      = None

