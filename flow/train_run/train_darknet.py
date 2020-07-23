# from comet_ml import ExistingExperiment, Experiment
# experiment = Experiment(api_key="48CqCtjT2NoeKaYI84VpShSe2",
#                         project_name="ice-seals", workspace="readicculus")
import mlflow
from mlflow import pytorch
from torch.utils.data import DataLoader
#
# experiment = ExistingExperiment(api_key="48CqCtjT2NoeKaYI84VpShSe2",
#                         project_name="ice-seals", workspace="readicculus", previous_experiment='a5de30255f9c44fb8b0180f7b64d4720')
# experiment.set
# experiment.clean()
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from PyTorchYOLOv4.models import Darknet, load_darknet_weights
from PyTorchYOLOv4.utils.DNConfig import DNConfig
from PyTorchYOLOv4.utils.DNData import DNData
from PyTorchYOLOv4.utils.utils import *
from PyTorchYOLOv4.test import test

import torch.distributed as dist

from PyTorchYOLOv4.utils.datasets import *
from PyTorchYOLOv4.utils.torch_utils import *
from flow import experiment as expt

AUG = True
# Hyperparameters
hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)  # TODO parse from config
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum # TODO parse from config
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

def train_darknet(dn_cfg, dn_data, weights, opt, device, mixed_precision):
    with mlflow.start_run(run_name='train_yolov4', experiment_id=expt.experiment_id):
        epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
        batch_size = opt.batch_size
        accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
        imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)


        # Image Sizes
        gs = 64  # (pixels) grid size
        assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
        opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
        if opt.multi_scale:
            if imgsz_min == imgsz_max:
                imgsz_min //= 1.5
                imgsz_max //= 0.667
            grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
            imgsz_min, imgsz_max = grid_min * gs, grid_max * gs
        img_size = imgsz_max  # initialize with max size

        for o in vars(opt):
            mlflow.set_tag('opt_%s' % str(o), str(vars(opt)[o]))

        # Configure run
        init_seeds()

        train_path = dn_data.f_train
        test_path = dn_data.f_test
        nc = 1 if opt.single_cls else int(dn_data.n_classes)  # number of classes
        hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset


        print("DEVICE: %s" % device.type)


        try:  # Mixed precision training https://github.com/NVIDIA/apex
            from apex import amp
        except:
            print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
            mixed_precision = False  # not installed


        mlflow.log_artifact(dn_cfg.cfg_path)
        mlflow.set_tag('pretrained_weights_path', weights)

        for h in hyp:
            mlflow.log_param('hyp_%s'%h, str(hyp[h]))

        ##################### load the model
        model = Darknet(dn_cfg, verbose=True).to(device)
        # model.dn_cfg.comet_log_params(experiment)

        # Optimizer
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(model.named_parameters()).items():
            if '.bias' in k:
                pg2 += [v]  # biases
            elif 'Conv2d.weight' in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else

        if opt.adam:
            # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
            optimizer = optim.Adam(pg0, lr=hyp['lr0'])
            # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
        else:
            optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)


        optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        del pg0, pg1, pg2
        start_epoch = 0
        best_fitness = 0.0
        if weights.endswith('.pt'): # pytorch format
            load_torch_weights(weights, model, optimizer, device)
        elif len(weights) > 0:  # darknet format
            load_darknet_weights(model, weights)
        else:
            initialize_weights(model)

        # Mixed precision training https://github.com/NVIDIA/apex
        if mixed_precision:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scheduler.last_epoch = start_epoch - 1  # see link below
        # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822

        # Initialize distributed training
        if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
            dist.init_process_group(backend='nccl',  # 'distributed backend'
                                    init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                    world_size=1,  # number of nodes for distributed training
                                    rank=0)  # distributed training node rank
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

        # Dataset
        dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                      augment=AUG,
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=opt.rect,  # rectangular training
                                      cache_images=opt.cache_images,
                                      single_cls=opt.single_cls)

        # Dataloader
        batch_size = min(batch_size, len(dataset))
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

        # nw = 0
        dataloader =DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=nw,
                                                 shuffle=not opt.rect,
                                                 # Shuffle=True unless rectangular training is used
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn)

        # Testloader
        testloader = DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                     hyp=hyp,
                                                                     rect=True,
                                                                     cache_images=opt.cache_images,
                                                                     single_cls=opt.single_cls),
                                                 batch_size=batch_size,
                                                 num_workers=nw,
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn)

        # Model parameters
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights

        # Model EMA
        ema = ModelEMA(model)

        # Start training
        nb = len(dataloader)  # number of batches
        n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
        maps = np.zeros(nc)  # mAP per class
        # torch.autograd.set_detect_anomaly(True)
        results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        t0 = time.time()
        print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
        print('Using %g dataloader workers' % nw)
        print('Starting training for %g epochs...' % epochs)
        step = 0

        for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
            model.train()

            # Update image weights (optional)
            if dataset.image_weights:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(range(dataset.n), weights=image_weights,
                                                 k=dataset.n)  # rand weighted idx

            mloss = torch.zeros(4).to(device)  # mean losses
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
            for i, (
            imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
                targets = targets.to(device)
                # Burn-in
                if ni <= n_burn * 2:
                    model.gr = np.interp(ni, [0, n_burn * 2], [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                    if ni == n_burn:  # burnin complete
                        print_model_biases(model)

                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, [0, n_burn], [0.9, hyp['momentum']])

                # Multi-Scale
                if opt.multi_scale:
                    if ni / accumulate % 1 == 0:  # adjust img_size (67% - 150%) every 1 batch
                        img_size = random.randrange(grid_min, grid_max + 1) * gs
                    sf = img_size / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in
                              imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                pred = model(imgs)

                # Loss
                loss, loss_items = compute_loss(pred, targets, model)
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                # Backward
                loss *= batch_size / 64  # scale loss
                if mixed_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Optimize
                if ni % accumulate == 0:
                    step = int(ni/accumulate)
                    # equivalent of an iteration in darknet
                    optimizer.step()
                    optimizer.zero_grad()
                    ema.update(model)

                # Print
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  #
                s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
                if ni % accumulate == 0:
                    mloss_val = mloss.tolist()
                    mlflow.log_metric('GIoU', mloss_val[0], step=step)
                    mlflow.log_metric('obj', mloss_val[1], step=step)
                    mlflow.log_metric('cls', mloss_val[2], step=step)
                    mlflow.log_metric('total', mloss_val[3], step=step)
                    mlflow.log_metric('targets', len(targets))
                    mlflow.log_metric('img_size', img_size)

                pbar.set_description(s)

                # Plot
                if ni < 3:
                    f = 'train_batch%g.png' % i  # filename
                    log_image_bbox_artifact(imgs=imgs, targets=targets, ni=ni, name='train')
                    # if tb_writer:
                    #     tb_writer.add_image(f, cv2.imread(f)[:, :, ::-1], dataformats='HWC')
                    #     # tb_writer.add_graph(model, imgs)  # add model to tensorboard

                # end batch ------------------------------------------------------------------------------------------------

            # Update scheduler
            scheduler.step()

            # Process epoch results
            ema.update_attr(model)
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                results, maps = test(dn_cfg,
                                          dn_data,
                                          batch_size=batch_size,
                                          img_size=imgsz_test,
                                          model=ema.ema,
                                          save_json=final_epoch and False,
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          epoch = epoch, ni=ni)
                results_val = list(results)
                mlflow.log_metric('Precision', results_val[0])
                mlflow.log_metric('Recall', results_val[1])
                mlflow.log_metric('mAP', results_val[2])
                mlflow.log_metric('mF1', results_val[3])
                mlflow.log_metric('test_GIoU',0 if np.isnan(results_val[4]) else results_val[4])
                mlflow.log_metric('test_obj',0 if np.isnan(results_val[5]) else results_val[5])
                mlflow.log_metric('test_cls',0 if np.isnan(results_val[6]) else results_val[6])

            # Write
            with open('results.txt', 'a') as f:
                f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
                # (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1].tolist()) + list(results), tags):
                mlflow.log_metric(tag, x, step=ni)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
            if fi > best_fitness:
                best_fitness = fi

            # log model as artifact --------
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                results_file = 'results.txt'
                with open(results_file, 'r') as f:  # create checkpoint
                    chkpt = {'epoch': epoch,
                             'best_fitness': best_fitness,
                             'training_results': f.read(),
                             'model': ema.ema.module.state_dict() if hasattr(model,
                                                                             'module') else ema.ema.state_dict(),
                             'optimizer': None if final_epoch else optimizer.state_dict()}

                # Save last, best and delete
                print('Logging model to mlflow')
                if (best_fitness == fi) and not final_epoch and not epoch == 0:
                    best_name = 'best_%d' % epoch
                    print('new BEST!')
                    local_path = s3_cache.save_ckpt(chkpt, "checkpoints", best_name+'.pt')
                    mlflow.log_artifact(local_path, 'checkpoints')
                    # mlflow.pytorch.log_model(model, os.path.join('models', best_name))
                else:
                    name = 'epoch_%d' % epoch
                    s3_cache.save_ckpt(chkpt, "checkpoints", name+'.pt')
                del chkpt

            # end epoch ----------------------------------------------------------------------------------------------------

        # end training
        # n = opt.name
        # if len(n):
        #     n = '_' + n if not n.isnumeric() else n
        #     fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        #     for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
        #         if os.path.exists(f1):
        #             os.rename(f1, f2)  # rename
        #             ispt = f2.endswith('.pt')  # is *.pt
        #             strip_optimizer(f2) if ispt else None  # strip optimizer
        #             os.system(
        #                 'gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload




if __name__ == '__main__':
    mixed_precision = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=2)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, default='cfg/yolov4-pacsp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 832], help='[min_train, max-train, test]')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov4-pacsp.pt', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    opt = parser.parse_args()
    # opt.weights = last if opt.resume else opt.weights
    print(opt)
    if hyp['fl_gamma']:
        print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])

    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
    device = select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    dn_data = DNData()
    dn_data.load(opt.data)

    dn_cfg = DNConfig(opt.cfg)
    if not opt.evolve:  # Train normally
        train_darknet(dn_cfg=dn_cfg,
                      dn_data=dn_data,
                      weights='',
                      opt=opt,
                      device=device,
                      mixed_precision=mixed_precision)