import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import itertools
import struct # get_image_size
import imghdr # get_image_size

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = min(box1[0], box2[0])
        x2_max = max(box1[2], box2[2])
        y1_min = min(box1[1], box2[1])
        y2_max = max(box1[3], box2[3])
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    else:
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        x1_min = min(box1[0]-w1/2.0, box2[0]-w2/2.0)
        x2_max = max(box1[0]+w1/2.0, box2[0]+w2/2.0)
        y1_min = min(box1[1]-h1/2.0, box2[1]-h2/2.0)
        y2_max = max(box1[1]+h1/2.0, box2[1]+h2/2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    carea = 0
    if w_cross <= 0 or h_cross <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    uarea = area1 + area2 - carea
    return float(carea/uarea)

def multi_bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = torch.min(boxes1[0], boxes2[0])
        x2_max = torch.max(boxes1[2], boxes2[2])
        y1_min = torch.min(boxes1[1], boxes2[1])
        y2_max = torch.max(boxes1[3], boxes2[3])
        w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
        w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
    else:
        w1, h1 = boxes1[2], boxes1[3]
        w2, h2 = boxes2[2], boxes2[3]
        x1_min = torch.min(boxes1[0]-w1/2.0, boxes2[0]-w2/2.0)
        x2_max = torch.max(boxes1[0]+w1/2.0, boxes2[0]+w2/2.0)
        y1_min = torch.min(boxes1[1]-h1/2.0, boxes2[1]-h2/2.0)
        y2_max = torch.max(boxes1[1]+h1/2.0, boxes2[1]+h2/2.0)
    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    mask = (((w_cross <= 0) + (h_cross <= 0)) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea

def multi_max_bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = torch.min(boxes1[0], boxes2[0])
        x2_max = torch.max(boxes1[2], boxes2[2])
        y1_min = torch.min(boxes1[1], boxes2[1])
        y2_max = torch.max(boxes1[3], boxes2[3])
        w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
        w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
    else:
        w1, h1 = boxes1[2], boxes1[3]
        w2, h2 = boxes2[2], boxes2[3]
        x1_min = torch.min(boxes1[0]-w1/2.0, boxes2[0]-w2/2.0)
        x2_max = torch.max(boxes1[0]+w1/2.0, boxes2[0]+w2/2.0)
        y1_min = torch.min(boxes1[1]-h1/2.0, boxes2[1]-h2/2.0)
        y2_max = torch.max(boxes1[1]+h1/2.0, boxes2[1]+h2/2.0)
    x1 = torch.min(boxes1[0]-w1/2.0, boxes1[0]-w1/2.0)
    x2 = torch.min(boxes1[0]+w1/2.0, boxes1[0]+w1/2.0)

    y1 = torch.min(boxes1[1] - h1 / 2.0, boxes1[1] - h1 / 2.0)
    y2 = torch.min(boxes1[1] + h1 / 2.0, boxes1[1] + h1 / 2.0)
    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    mask = (((w_cross <= 0) + (h_cross <= 0)) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    x = (x1_min == x1)*(x2_max == x2)*(y1_min == y1)*(y2_max == y2)
    mask = ((mask+x) > 1)
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def grid_nms(boxes, nms_thresh):

    grid_size = [52, 26, 13]
    for n in range(3):
        boxes_tmp = []
        if n == 0:
            for i in range(len(boxes)):
                boxes_tmp.append(boxes[i])
        else:
            for i in range(len(grid_boxes)):
                boxes_tmp.append(grid_boxes[i])
        grid = [[[] for i in range(grid_size[n])] for i in range(grid_size[n])]
        grid_boxes = []
        for i in range(len(boxes_tmp)):
            x = int(boxes_tmp[i][0]*grid_size[n])
            y = int(boxes_tmp[i][1]*grid_size[n])
            grid[x][y].append(boxes_tmp[i])
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if len(grid[i][j]) > 0:
                    box = grid[i][j][0]
                    if len(grid[i][j]) > 1:
                        box = nms(grid[i][j], nms_thresh)
                        grid_boxes += box
                        # print("iou", i, j, len(grid[i][j]), len(box))
                    else:
                        grid_boxes.append(box)

    return grid_boxes

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]                

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                iou = bbox_iou(box_i, box_j, x1y1x2y2=False)
                # print(j, iou)
                if iou > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0

    return out_boxes

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def get_all_boxes(output, conf_thresh, num_classes, only_objectness=0, validation=False, use_cuda=True):
    # total number of inputs (batch size)
    # first element (x) for first tuple (x, anchor_mask, num_anchor)
    tot = output[0]['x'].data.size(0)
    all_boxes = [[] for i in range(tot)]
    for i in range(len(output)):
        pred = output[i]['x'].data

        # find number of workers (.s.t, number of GPUS) 
        nw = output[i]['n'].data.size(0)
        anchors = output[i]['a'].chunk(nw)[0]
        num_anchors = output[i]['n'].data[0].item()

        b = get_region_boxes(pred, conf_thresh, num_classes, anchors, num_anchors, \
                only_objectness=only_objectness, validation=validation, use_cuda=use_cuda)
        for t in range(tot):
            all_boxes[t] += b[t]
    return all_boxes

def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False, use_cuda=True):
    device = torch.device("cuda" if use_cuda else "cpu")
    num_props = 13 #13
    anchors = anchors.to(device)
    anchor_step = anchors.size(0)//num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes+num_props)*num_anchors)
    h = output.size(2)
    w = output.size(3)
    cls_anchor_dim = batch*num_anchors*h*w

    t0 = time.time()
    all_boxes = []
    output = output.view(batch*num_anchors, 5+num_classes+num_props, h*w).transpose(0,1).contiguous().view(5+num_classes+num_props, cls_anchor_dim)

    grid_x = torch.linspace(0, w-1, w).repeat(batch*num_anchors, h, 1).view(cls_anchor_dim).to(device)
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(cls_anchor_dim).to(device)
    ix = torch.LongTensor(range(0,2)).to(device)
    anchor_w = anchors.view(num_anchors, anchor_step).index_select(1, ix[0]).repeat(1, batch, h*w).view(cls_anchor_dim)
    anchor_h = anchors.view(num_anchors, anchor_step).index_select(1, ix[1]).repeat(1, batch, h*w).view(cls_anchor_dim)

    xs, ys = torch.sigmoid(output[0]) + grid_x, torch.sigmoid(output[1]) + grid_y
    ws, hs = torch.exp(output[2]) * anchor_w.detach(), torch.exp(output[3]) * anchor_h.detach()
    det_confs = torch.sigmoid(output[4])

    prop = output[5+num_classes:5+num_classes+num_props].transpose(0, 1).sigmoid()

    # by ysyun, dim=1 means input is 2D or even dimension else dim=0
    cls_confs = torch.nn.Softmax(dim=1)(output[5:5+num_classes].transpose(0,1)).detach()
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()

    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs, ys = convert2cpu(xs), convert2cpu(ys)
    ws, hs = convert2cpu(ws), convert2cpu(hs)
    props = convert2cpu(prop)

    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    nn = [0, 0, 0]
    t2 = time.time()
    for b in range(batch):
        boxes = []
        # print(w, h)

        for cy in range(h):
            for cx in range(w):
                boxes_anchors = []
                for i in range(num_anchors):

                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        # print('conf is: ', conf)
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        bpro = props[ind]

                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id, bpro]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
                        # boxes_anchors.append(box)
                        nn[i] += 1
                # boxes_anchors = nms(boxes_anchors, 0.1)
                # boxes += boxes_anchors
        print(w, nn)
        all_boxes.append(boxes)
    t3 = time.time()

    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return all_boxes

def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])

    INFO = [
        ['male', 0, 1],
        ['female', 0, 1],

        ['back_head', 0, 1],#2
        ['side_face', 0, 1],
        ['frontal_face', 0, 1],

        ['childhood', 0, 1],#5
        ['juvenile', 0, 1],
        ['youth', 0, 1],
        ['middle', 0, 1],
        ['agedness', 0, 1],

        ['none_glasses', 0, 1],#10
        ['common_glasses', 0, 1],
        ['sun_glasses', 0, 1],
    ]

    # prop_list = [['Female', 'Male'], ['Customer', 'Stuff'], ['Stand', 'Sit'], ['With phone', 'Without phone']]
    prop_list = [['head', 'side', 'face', 'ff'], ['unknown', 'male', 'female'], ['unknown', 'none', 'common', 'sun'], ['unknown', 'yellow', 'white', 'black', 'arabs'], ['unknown', 'childhood', 'juvenile', 'youth', 'middle', 'agedness']]
    prop_thresh = 0.5
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.shape[1]
    height = img.shape[0]
    dst_boxes = []
    ccc = (220, 120, 0)

    for j in range(len(boxes)):
        person = {}
        cct = (120, 0, 240)
        box = boxes[j]
        x1 = int(round((box[0] - box[2]/2.0) * width))
        y1 = int(round((box[1] - box[3]/2.0) * height))
        x2 = int(round((box[0] + box[2]/2.0) * width))
        y2 = int(round((box[1] + box[3]/2.0) * height))
        props = box[7]
        wbox = [x1, y1, x2, y2]
        rbox = [float(box[4])]
        rbox.extend(box[7].tolist())
        rbox = [round(z, 5) for z in rbox]
        wbox.extend(rbox)
        dst_boxes.append(wbox)
        # gender, customer, stand, play_with_phone

        d_prop = []

        if props[2] > 0.5:
            cct = (120, 120, 120)
        elif props[3] > 0.5:
            cct = (0, 250, 250)
        elif props[4] > 0.5:
            cct = (220, 120, 0)
            if props[0] > 0.5:
                cct = (220, 120, 0)
                person['gender'] = 'male'
            elif props[1] > 0.3:
                cct = (250, 0, 250)
                person['gender'] = 'female'
            else:
                cct = (0, 0, 250)
                person['gender'] = 'unknown'
            d_prop.append('gender' + ':' + person['gender'])

            if props[5] > 0.5:
                person['age'] = 'childhood'
            elif props[6] > 0.5:
                person['age'] = 'juvenile'
            elif props[7] > 0.5:
                person['age'] = 'youth'
            elif props[8] > 0.5:
                person['age'] = 'middle'
            elif props[9] > 0.5:
                person['age'] = 'agedness'
            else:
                person['age'] = 'unknown'
            d_prop.append('age' + ':' + person['age'])

            if props[10] > 0.5:
                person['glasses'] = 'none'
            elif props[11] > 0.5:
                person['glasses'] = 'common'
            elif props[12] > 0.5:
                person['glasses'] = 'sun'
            else:
                person['glasses'] = 'unknown'
            d_prop.append('glasses' + ':' + person['glasses'])
        else:
            cct = (250, 250, 250)

        # if props[0] > props[1]:
        #     d_prop.append(INFO[0][0] + ' ' + str(float(props[0])))
        # elif props[0] < props[1]:
        #     d_prop.append(INFO[1][0] + ' ' + str(float(props[1])))

        for i in range(len(props)):
            print(i, props[i])

            # prop_thresh = 0.5/(len(prop_list[i])-1)
            # id = 0
            # for n in range(len(prop_list[i])):
            #     step = prop_thresh + n*2*prop_thresh
            #     if props[i] < step:
            #         id = n
            #         break
            # if i == 0:
            #     if props[i] < 0.25:
            #         ccc = (120, 120, 120)
            #         break
            #     elif props[i] < 0.75:
            #         ccc = (255, 255, 255)
            #         break
            #     continue
            # elif i == 1:
            #     if props[i] > 0.75:
            #         ccc = (255, 0, 255)
            # d_prop.append(prop_list[i][id])

            # if props[i] >= prop_thresh:
            #     d_prop.append(prop_list[i][0])
            # else:
            #     d_prop.append(prop_list[i][1])
            #     if i == 1:
            #         cct = (180, 180, 20)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            #print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            namep = ','.join(d_prop)
            t_wi = abs(x2 - x1)
            t_size = 0.7
            if t_wi < 200:
                t_size = 0.5
            elif t_wi >= 200 and t_wi < 280:
                t_size ==0.6

            for i in range(len(d_prop)):
                img = cv2.putText(img, d_prop[i], (x2, y1 + i*14), cv2.FONT_HERSHEY_SIMPLEX, t_size, cct,1)
            # img = cv2.putText(img, namep, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, t_size, cct, 2)
            # if len(d_prop) > 0:
            #     img = cv2.putText(img, d_prop[0], (x1-8*len(d_prop[0]), y1), cv2.FONT_HERSHEY_SIMPLEX, t_size, cct, 1)
            #     img = cv2.putText(img, d_prop[1], (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, t_size, cct, 1)
            #     img = cv2.putText(img, d_prop[2], (x1-8*len(d_prop[2]), y2), cv2.FONT_HERSHEY_SIMPLEX, t_size, cct, 1)
            #     img = cv2.putText(img, d_prop[3], (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, t_size, cct, 1)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), cct, 2)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img, dst_boxes

def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    print("%d box(es) is(are) found" % len(boxes))
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline=rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img

def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size//5, 5) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'rb') as fp:
        lines = fp.readlines()
    for line in lines:
        class_names.append(line.strip())
    return class_names

def image2torch(img):
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray: # cv2 image
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknown image type")
        exit(-1)
    return img

import types
def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=True):
    model.eval()
    t0 = time.time()
    img = image2torch(img)
    t1 = time.time()

    img = img.to(torch.device("cuda" if use_cuda else "cpu"))
    t2 = time.time()

    out_boxes = model(img)
    boxes = get_all_boxes(out_boxes, conf_thresh, model.num_classes, use_cuda=use_cuda)[0]
    
    t3 = time.time()
    # print(len(boxes))
    boxes = grid_nms(boxes, nms_thresh)
    # print('end', len(boxes))
    # boxes = nms(boxes, nms_thresh)
    t4 = time.time()

    if True:
        print('-----------------------------------')
        print(' image to tensor : %f' % (t1 - t0))
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('             nms : %f' % (t4 - t3))
        print('           total : %f' % (t4 - t0))
        print('-----------------------------------')
    return boxes

def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options

def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets
      
def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count(b'\n')
    thefile.close( )
    return count

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24: 
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2 
                ftype = 0 
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2 
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))

def savelog(message):
    logging(message)
    with open('savelog.txt', 'a') as f:
        print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))
