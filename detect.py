import sys
import time
from PIL import Image, ImageDraw
#from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet

namesfile=None
def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    #m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    # if m.num_classes == 20:
    #     namesfile = 'data/voc.names'
    # elif m.num_classes == 80:
    #     namesfile = 'data/coco.names'
    # else:
    #     namesfile = 'data/names'
    
    use_cuda = True
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))
    
    #for i in range(2):
    start = time.time()
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    finish = time.time()
        #if i == 1:
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)

# def detect_cv2(cfgfile, weightfile, imgfile, savepath):
def detect_cv2(m, imgfile, savepath):
    import cv2


    # if m.num_classes == 20:
    #     namesfile = 'data/voc.names'
    # elif m.num_classes == 80:
    #     namesfile = 'data/coco.names'
    # else:
    #     namesfile = 'data/names'
    
    use_cuda = True
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.1, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    _, ds_boxes = plot_boxes_cv2(img, boxes, savename=savepath, class_names=class_names)
    return ds_boxes

def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = True
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def write_js(boxes, name):
    dst = {}
    dst['image_id'] = name
    dst['object'] = []
    for b in boxes:
        box = {}
        box['confidence'] = round(b[4], 5)
        box["minx"] = b[0]
        box["miny"] = b[1]
        box["maxx"] = b[2]
        box["maxy"] = b[3]
        box["staff"] = float('%.2f'%(float(round(1.0 - b[6], 5))))
        box["customer"] = float('%.2f'%(float(round(b[6], 5))))
        box["stand"] = float('%.2f'%(float(round(b[7], 5))))
        box["sit"] = float('%.2f'%(float(round(1.0 - b[7], 5))))
        box["play_with_phone"] = float('%.2f'%(float(round(b[8], 5))))
        box["male"] = float('%.2f'%(float(round(1.0 - b[5], 5))))
        box["female"] = float('%.2f'%(float(round(b[5], 5))))
        dst['object'].append(box)

    return dst

if __name__ == '__main__':
    import json

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    globals()["namesfile"] = ''
    cfgfile = './cfg/yoloFaceTest.cfg'
    weightfile = './models/f4_star015000.weights'
    # weightfile = './models/yolov3-face_100000.weights'
    # weightfile = './models/pul_models/ap000450.weights'
    # weightfile = './models/nnp000310.weights'
    imgfile = './data/person.jpg'
    label_path = './data/coco.names'
    globals()["namesfile"] = label_path

    mdir = '/opt/RedOctober/data/faceTest/'
    src_dir = mdir + 'test.txt'
    import cv2

    m = Darknet(cfgfile)

    # m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    results = {}
    results['results'] = []
    # js_path = 'results_s.json'
    js_path = 'results_standard.json'

    with open('f2id.json', 'r') as fr:
        jid = json.load(fr)

    with open(src_dir, 'rb') as ff:
        for i in ff:
            imgf = i[:-1]
            img_name = imgf.split('/')[-1]
            # sn = img_name.split('_')[1]
            # if int(sn) == 4:
            save_path = mdir + 'results/' + img_name
            boxes = detect_cv2(m, imgf, save_path)
            # ds_box = write_js(boxes, jid[img_name])
            # results['results'].append(ds_box)
    with open(js_path, 'wb') as fw:
        json.dump(results, fw)

