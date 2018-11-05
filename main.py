import argparse
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import gc
import config

import dataset
from utils import *
from cfg import parse_cfg
from darknet import Darknet
import argparse

mFlags = None
mConfig = None
mModel = None

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cfg/yoloFace.data', help='data definition file')
    parser.add_argument('--config', '-c', type=str, default='cfg/yoloFace.cfg', help='network configuration file')
    # parser.add_argument('--weights', '-w', type=str, default='./models/cf9_star000700.weights', help='initial weights file')
    parser.add_argument('--weights', '-w', type=str, default='./models/darknet53.conv.74', help='initial weights file')
    parser.add_argument('--noeval', '-n', dest='no_eval', action='store_true', help='prohibit test evalulation')
    parser.add_argument('--reset', '-r', action="store_true", default=False, help='initialize the epoch and model seen value')
    parser.add_argument('--localmax', '-l', action="store_true", default=False, help='save net weights for local maximum fscore')
    flags, _ = parser.parse_known_args()
    return flags

def set_cfg():
    c = config.parameter()

    datacfg = mFlags.data
    c.cfgfile = mFlags.config
    c.weightfile = mFlags.weights
    no_eval = mFlags.no_eval

    data_options = read_data_cfg(datacfg)
    net_options = parse_cfg(c.cfgfile)[0]

    c.use_cuda = torch.cuda.is_available() and (True if c.use_cuda is None else c.use_cuda)
    c.trainlist = data_options['train']
    c.testlist = data_options['valid']
    c.backupdir = data_options['backup']
    c.gpus = data_options['gpus']
    c.ngpus = len(c.gpus.split(','))
    c.num_workers = int(data_options['num_workers'])

    c.batch_size = int(net_options['batch'])
    c.max_batches = 10 * int(net_options['max_batches'])
    c.learning_rate = float(net_options['learning_rate'])
    c.momentum = float(net_options['momentum'])
    c.decay = float(net_options['decay'])
    c.steps = [float(step) for step in net_options['steps'].split(',')]
    c.scales = [float(scale) for scale in net_options['scales'].split(',')]

    try:
        c.max_epochs = int(net_options['max_epochs'])
    except KeyError:
        nsamples = file_lines(c.trainlist)
        c.max_epochs = (c.max_batches * c.batch_size) // nsamples + 1
    seed = int(time.time())
    torch.manual_seed(seed)
    if c.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = c.gpus
        torch.cuda.manual_seed(seed)

    c.device = torch.device("cuda" if c.use_cuda else "cpu")
    print('set_cfg')

    return c

def loadLayer():
    cfgfile = mFlags.config
    weightfile = mFlags.weights
    model = Darknet(cfgfile, use_cuda=mConfig.use_cuda)
    model.load_weights(weightfile)

    nsamples = file_lines(mConfig.trainlist)
    # initialize the model
    if mFlags.reset:
        model.seen = 0
        mConfig.init_epoch = 0
    else:
        mConfig.init_epoch = model.seen // nsamples

    mConfig.loss_layers = model.loss_layers
    for l in mConfig.loss_layers:
        l.seen = model.seen


    if mConfig.use_cuda:
        if mConfig.ngpus > 1:
            model = torch.nn.DataParallel(model).to(mConfig.device)
        else:
            model = model.to(mConfig.device)

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': mConfig.decay * mConfig.batch_size}]

    mConfig.optimizer = optim.SGD(model.parameters(),
                                  lr=mConfig.learning_rate / mConfig.batch_size, momentum=mConfig.momentum,
                                  dampening=0, weight_decay=mConfig.decay * mConfig.batch_size)

    return model

def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = mConfig.learning_rate
    for i in range(len(mConfig.steps)):
        scale = mConfig.scales[i] if i < len(mConfig.scales) else 1
        if batch >= mConfig.steps[i]:
            lr = lr * scale
            if batch == mConfig.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/mConfig.batch_size
    return lr

def curmodel():
    if mConfig.ngpus > 1:
        cur_model = mModel.module
    else:
        cur_model = mModel
    return cur_model


def loadData():
    cur_model = curmodel()
    listDataset = dataset.listDataset(mConfig.trainlist, shape=(cur_model.width, cur_model.height),
                        shuffle=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]),
                        train=True,
                        seen=cur_model.seen,
                        batch_size=mConfig.batch_size,
                        num_workers=mConfig.num_workers)
    return listDataset

def train(epoch):
    t0 = time.time()
    cur_model = curmodel()
    init_width = cur_model.width
    init_height = cur_model.height
    kwargs = {'num_workers': mConfig.num_workers, 'pin_memory': True} if mConfig.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(loadData(), batch_size=mConfig.batch_size, shuffle=False, **kwargs)

    processed_batches = cur_model.seen // mConfig.batch_size
    lr = adjust_learning_rate(mConfig.optimizer, processed_batches)
    logging('[%03d] processed %d samples, lr %e' % (epoch, epoch * len(train_loader.dataset), lr))
    mModel.train()
    t1 = time.time()
    avg_time = torch.zeros(9)
    for batch_idx, (data, target) in enumerate(train_loader):
        t2 = time.time()
        adjust_learning_rate(mConfig.optimizer, processed_batches)
        processed_batches = processed_batches + 1
        # if (batch_idx+1) % dot_interval == 0:
        #    sys.stdout.write('.')

        t3 = time.time()
        data, target = data.to(mConfig.device), target.to(mConfig.device)

        t4 = time.time()
        mConfig.optimizer.zero_grad()

        t5 = time.time()
        output = mModel(data)

        t6 = time.time()
        org_loss = []
        for i, l in enumerate(mConfig.loss_layers):
            l.seen = l.seen + data.data.size(0)
            ol = l(output[i]['x'], target)
            org_loss.append(ol)

        t7 = time.time()

        # for i, l in enumerate(reversed(org_loss)):
        #    l.backward(retain_graph=True if i < len(org_loss)-1 else False)
        # org_loss.reverse()
        sum(org_loss).backward()

        nn.utils.clip_grad_norm_(mModel.parameters(), 10000)
        # for p in model.parameters():
        #    p.data.add_(-lr, p.grad.data)

        t8 = time.time()
        mConfig.optimizer.step()

        t9 = time.time()
        if False and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2 - t1)
            avg_time[1] = avg_time[1] + (t3 - t2)
            avg_time[2] = avg_time[2] + (t4 - t3)
            avg_time[3] = avg_time[3] + (t5 - t4)
            avg_time[4] = avg_time[4] + (t6 - t5)
            avg_time[5] = avg_time[5] + (t7 - t6)
            avg_time[6] = avg_time[6] + (t8 - t7)
            avg_time[7] = avg_time[7] + (t9 - t8)
            avg_time[8] = avg_time[8] + (t9 - t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0] / (batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1] / (batch_idx)))
            print('cuda to variable : %f' % (avg_time[2] / (batch_idx)))
            print('       zero_grad : %f' % (avg_time[3] / (batch_idx)))
            print(' forward feature : %f' % (avg_time[4] / (batch_idx)))
            print('    forward loss : %f' % (avg_time[5] / (batch_idx)))
            print('        backward : %f' % (avg_time[6] / (batch_idx)))
            print('            step : %f' % (avg_time[7] / (batch_idx)))
            print('           total : %f' % (avg_time[8] / (batch_idx)))
        t1 = time.time()
        del data, target
        org_loss = []
        gc.collect()

    print('')
    t1 = time.time()
    nsamples = len(train_loader.dataset)
    logging('training with %f samples/s' % (nsamples / (t1 - t0)))
    return nsamples


def savemodel(epoch, nsamples, curmax=False):
    cur_model = curmodel()
    if curmax:
        logging('save local maximum weights to %s/localmax.weights' % (mConfig.backupdir))
    else:
        logging('save weights to %s/%06d.weights' % (mConfig.backupdir, epoch))
    cur_model.seen = epoch * nsamples
    if curmax:
        cur_model.save_weights('%s/localmax.weights' % (mConfig.backupdir))
    else:
        cur_model.save_weights('%s/f1_star%06d.weights' % (mConfig.backupdir, epoch))
        cur_model.save_weights('%s/f1.weights' % (mConfig.backupdir))
        # cur_model.save_weights('%s/pul_models/ap%06d.weights' % (backupdir, epoch))
        old_wgts = 'npz%s/ap%06d.weights' % (mConfig.backupdir, epoch - mConfig.keep_backup * mConfig.save_interval)
        try:  # it avoids the unnecessary call to os.path.exists()
            os.remove(old_wgts)
        except OSError:
            pass


def loop():
    try:
        print("Training for ({:d},{:d})".format(mConfig.init_epoch, mConfig.max_epochs))
        fscore = 0
        if not mConfig.no_eval and mConfig.init_epoch > mConfig.test_interval:
            print('>> initial evaluating ...')
            # mfscore = test(init_epoch)
            print('>> done evaluation.')
        else:
            mfscore = 0.5
        for epoch in range(mConfig.init_epoch + 1, mConfig.max_epochs):
            nsamples = train(epoch)
            if not mConfig.no_eval and epoch > mConfig.test_interval and (epoch % mConfig.test_interval) == 0:
                print('>> intermittent evaluating ...')
                # fscore = test(epoch)
                print('>> done evaluation.')
            if epoch % mConfig.save_interval == 0:
                savemodel(epoch, nsamples)
                pass
            if mFlags.localmax and fscore > mfscore:
                mfscore = fscore
                savemodel(epoch, nsamples, True)
            print('-' * 90)
    except KeyboardInterrupt:
        print('=' * 80)
        print('Exiting from training by interrupt')

if __name__ == '__main__':
    mFlags = get_cfg()
    mConfig = set_cfg()
    mModel = loadLayer()
    loadData()
    loop()
    # test()
    print('end')