import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torchvision import transforms
from ShapeNetLoader import ShapeNetPart
import d_utils
import argparse
import random
from GEConvNet import GEConvNet_partseg

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='Shape Part Segmentation Training')
parser.add_argument('--workers', type=int, default=4, metavar='wrkers',help='number of workers, default=4')
parser.add_argument('--num_points', type=int, default=2048, help='num of points to use')
parser.add_argument('--num_classes', type=int, default=50, help='number of parts')
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',help='Size of batch')
parser.add_argument('--eval', type=bool, default=False, help='evaluate the model')
parser.add_argument('--base_lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--lr_clip', type=float, default=0.00001, metavar='LRclip', help='learning rate clip')
parser.add_argument('--lr_decay', type=float, default=0.75, metavar='LRdecay', help='learning rate decay')
parser.add_argument('--decay_step', type=int, default=50, metavar='decay_step', help='decay step')
parser.add_argument('--epochs', type=int, default=500, metavar='N',help='number of epochs')
parser.add_argument('--weight_decay', type=float, default=0, metavar='WD', help='weight decay')
parser.add_argument('--bn_momentum', type=float, default=0.9, metavar='BNM', help='bn momentum')
parser.add_argument('--bnm_clip', type=float, default=0.01, metavar='BNMc', help='bn momentum clip')
parser.add_argument('--bn_decay', type=float, default=0.5, metavar='BND', help='bn decay')
parser.add_argument('--print_freq_iter', type=int, default=100, help='requency in iteration for printing infomation')
parser.add_argument('--use_global', type=bool, default=False, help='use global coordinate system')
parser.add_argument('--with_normal', type=bool, default=True, help='use surface normal for pdr')
parser.add_argument('--save_path', type=str, default='PartSegno_with_n_ZSO3', metavar='N', help='Name of the experiment')
parser.add_argument('--model_path', type=str, default='pretrained/part_seg.pth', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--data_root', type=str, default='data/shapenetcore_partanno_segmentation_benchmark_v0_normal', metavar='data_root', help='data path')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

    print("\n**************************\n")

    try:
        os.makedirs(args.save_path)
    except OSError:
        pass

    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    train_dataset = ShapeNetPart(root=args.data_root, num_points=args.num_points, split='trainval', normalize=True,
                                 transforms=train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True
    )

    global test_dataset
    test_dataset = ShapeNetPart(root=args.data_root, num_points=args.num_points, split='test', normalize=True,
                                transforms=test_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True
    )

    model = GEConvNet_partseg(args, 50).to(device)

    # if torch.cuda.device_count() > 1:
    # model = nn.DataParallel(model) #enabling data parallelism
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    lr_lbmd = lambda e: max(args.lr_decay ** (e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay ** (e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = d_utils.BNMomentumScheduler(model, bnm_lmbd)


    criterion = nn.CrossEntropyLoss()
    num_batch = len(train_dataset) / args.batch_size

    # training
    train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)


def train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch):
    #PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()  # initialize augmentation
    global Class_mIoU, Inst_mIoU, ov_instance_ious
    Class_mIoU, Inst_mIoU, ov_instance_ious = 0, 0, 0
    batch_count = 0
    model.train()
    for epoch in range(args.epochs):
        for i, data in enumerate(train_dataloader, 0):
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
                bnm_scheduler.step(epoch - 1)
            points, normals, target, cls = data

            points, normals = d_utils.rotate_point_cloud(points, normals)
            points, normals = torch.from_numpy(points), torch.from_numpy(normals)
            points, normals, target = points.to(device), normals.to(device), target.to(device)
            points, normals, target = Variable(points), Variable(normals), Variable(target)

            # augmentation
            #points.data = PointcloudScaleAndTranslate(points.data)

            optimizer.zero_grad()

            batch_one_hot_cls = np.zeros((len(cls), 16))  # 16 object classes
            for b in range(len(cls)):
                batch_one_hot_cls[b, int(cls[b])] = 1
            batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls)
            batch_one_hot_cls = Variable(batch_one_hot_cls.float().to(device))

            pred = model(points.transpose(2, 1), batch_one_hot_cls,  normals.transpose(2, 1))

            pred = pred.reshape(-1, args.num_classes)
            target = target.view(-1, 1)[:, 0]
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            if i % args.print_freq_iter == 0:
                print('[epoch %3d: %3d/%3d] \t train loss: %0.6f \t lr: %0.5f' % (
                epoch + 1, i, num_batch, loss.data.clone(), lr_scheduler.get_lr()[0]))
            batch_count += 1

            # validation in between an epoch
            # if (epoch < 3 or epoch > 40) and args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
        with torch.no_grad():
            validate(test_dataloader, model, criterion, args, batch_count)


def validate(test_dataloader, model, criterion, args, iter):
    global Class_mIoU, Inst_mIoU, ov_instance_ious, test_dataset
    model.eval()

    seg_classes = test_dataset.seg_classes
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    losses = []
    for _, data in enumerate(test_dataloader, 0):
        points, normals, target, cls = data
        points, normals = d_utils.rotate_point_cloud_so3_norm(points, normals)
        points, normals = torch.from_numpy(points), torch.from_numpy(normals)
        points, normals, target = Variable(points, volatile=True), Variable(normals, volatile=True), Variable(target,
                                                                                                              volatile=True)
        points, normals, target = points.to(device), normals.to(device), target.to(device)

        batch_one_hot_cls = np.zeros((len(cls), 16))  # 16 object classes
        for b in range(len(cls)):
            batch_one_hot_cls[b, int(cls[b])] = 1
        batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls)
        batch_one_hot_cls = Variable(batch_one_hot_cls.float().to(device))

        pred = model(points.transpose(2, 1), batch_one_hot_cls, normals.transpose(2, 1))
        loss = criterion(pred.reshape(-1, args.num_classes), target.view(-1, 1)[:, 0])
        losses.append(loss.data.clone())
        pred = pred.data.cpu()
        target = target.data.cpu()
        pred_val = torch.zeros(len(cls), args.num_points).type(torch.LongTensor)
        # pred to the groundtruth classes (selected by seg_classes[cat])
        for b in range(len(cls)):
            cat = seg_label_to_cat[target[b, 0].item()]
            logits = pred[b, :, :]  # (num_points, num_classes)
            pred_val[b, :] = logits[:, seg_classes[cat]].max(1)[1] + seg_classes[cat][0]

        for b in range(len(cls)):
            segp = pred_val[b, :].numpy()
            segl = target[b, :].numpy()
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if np.sum((segl == l) | (segp == l)) == 0:
                    # part is not present in this shape
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))

    instance_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            instance_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    mean_class_ious = np.mean(list(shape_ious.values()))

    for cat in sorted(shape_ious.keys()):
        print('****** %s: %0.6f' % (cat, shape_ious[cat]))
    # print('************ Test Loss: %0.6f' % (np.array(losses).mean().item()))
    print('************ Class_mIoU: %0.6f' % (mean_class_ious))
    print('************ Instance_mIoU: %0.6f' % (np.mean(instance_ious)))

    if mean_class_ious > Class_mIoU or np.mean(instance_ious) > Inst_mIoU:
        if mean_class_ious > Class_mIoU:
            Class_mIoU = mean_class_ious
        if np.mean(instance_ious) > Inst_mIoU:
            Inst_mIoU = np.mean(instance_ious)
        print('saving new model....')
        torch.save(model.state_dict(), '%s/GEConvNetPartseglt_SO3_SO3_iter_%d_ins_%0.6f_cls_%0.6f.pth' % (
        args.save_path, iter, np.mean(instance_ious), mean_class_ious))
        print('BEST meanIOU = ', Class_mIoU, ' AND BEST instance IOU = ', Inst_mIoU)
        # torch.cuda.empty_cache()
    model.train()


from tqdm import tqdm
def test(args):
    model = GEConvNet_partseg(args, 50).to(device)
    model.load_state_dict(torch.load(args.model_path), strict=True)

    model.eval()

    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_dataset = ShapeNetPart(root=args.data_root, num_points=args.num_points, split='test', normalize=True,
                                transforms=test_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True
    )


    seg_classes = test_dataset.seg_classes
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    for _, data in enumerate(tqdm(test_dataloader), 0):
        points, normals, target, cls = data
        points, normals= d_utils.rotate_point_cloud_so3_norm(points, normals)
        points, normals = torch.from_numpy(points), torch.from_numpy(normals)
        points, normals, target = points.to(device), normals.to(device), target.to(device)

        batch_one_hot_cls = np.zeros((len(cls), 16))  # 16 object classes
        for b in range(len(cls)):
            batch_one_hot_cls[b, int(cls[b])] = 1
        batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls)
        batch_one_hot_cls = Variable(batch_one_hot_cls.float().to(device))

        pred = model(points.transpose(2, 1), batch_one_hot_cls, normals.transpose(2, 1))

        pred = pred.data.cpu()
        target = target.data.cpu()
        pred_val = torch.zeros(len(cls), args.num_points).type(torch.LongTensor)
        # pred to the groundtruth classes (selected by seg_classes[cat])
        for b in range(len(cls)):
            cat = seg_label_to_cat[target[b, 0].item()]
            logits = pred[b, :, :]  # (num_points, num_classes)
            pred_val[b, :] = logits[:, seg_classes[cat]].max(1)[1] + seg_classes[cat][0]

        for b in range(len(cls)):
            segp = pred_val[b, :].numpy()
            segl = target[b, :].numpy()
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if np.sum((segl == l) | (segp == l)) == 0:
                    # part is not present in this shape
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))

    instance_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            instance_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    mean_class_ious = np.mean(list(shape_ious.values()))

    for cat in sorted(shape_ious.keys()):
        print('****** %s: %0.6f' % (cat, shape_ious[cat]))
    # print('************ Test Loss: %0.6f' % (np.array(losses).mean().item()))
    print('************ Class_mIoU: %0.6f' % (mean_class_ious))
    print('************ Instance_mIoU: %0.6f' % (np.mean(instance_ious)))



if __name__ == "__main__":

    args = parser.parse_args()

    if not args.eval:
        main(args)
    else:
        test(args)

