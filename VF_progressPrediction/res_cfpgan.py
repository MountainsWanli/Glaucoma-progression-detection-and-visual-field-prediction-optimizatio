import json
import os
import shutil

import matplotlib.pyplot as plt
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter
from sklearn.metrics import classification_report
from timm.data.mixup import Mixup
from torchvision import datasets
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
from ema_pytorch import EMA
import timm
from sklearn.metrics import roc_auc_score, average_precision_score

# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    model.train()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        if len(data) % 2 != 0:
            if len(data) < 2:
                continue
            data = data[0:len(data) - 1]
            target = target[0:len(target) - 1]
            print(len(data))
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        # samples, targets = mixup_fn(data, target)
        output = model(data)
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast():
                loss = criterion_train(output, target)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
            if use_ema and epoch%ema_epoch==0:
                ema.update()
        else:
            loss = criterion_train(output, target)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            if use_ema and epoch%ema_epoch==0:
                ema.update()
        torch.cuda.synchronize()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss_meter.update(loss.item(), target.size(0))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), lr))
    ave_loss =loss_meter.avg
    acc = acc1_meter.avg
    print('epoch:{}\tloss:{:.2f}\tacc:{:.2f}'.format(epoch, ave_loss, acc))
    return ave_loss, acc

# 验证过程
@torch.no_grad()
def val(model, device, test_loader):
    global Best_ACC
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    val_list = []
    pred_list = []
    prob_list = []

    if use_ema and epoch % ema_epoch == 0:
        ema.apply_shadow()

    for data, target in test_loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        loss = criterion_val(output, target)
        _, pred = torch.max(output.data, 1)

        val_list.extend(target.cpu().numpy())
        pred_list.extend(pred.cpu().numpy())
        prob = torch.softmax(output, dim=1)[:, 1]  # 正类的概率
        prob_list.extend(prob.detach().cpu().numpy())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

    if use_ema and epoch % ema_epoch == 0:
        ema.restore()

    acc = acc1_meter.avg
    roc_auc = roc_auc_score(val_list, prob_list)
    pr_auc = average_precision_score(val_list, prob_list)

    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tAcc5:{:.3f}%\tROC-AUC:{:.4f}\tPR-AUC:{:.4f}\n'.format(
        loss_meter.avg, acc, acc5_meter.avg, roc_auc, pr_auc))

    if acc > Best_ACC:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module, file_dir + "/model_" + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
            torch.save(model.module, file_dir + '/best.pth')
        else:
            torch.save(model, file_dir + "/model_" + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
            torch.save(model, file_dir + '/best.pth')
        Best_ACC = acc

    return val_list, pred_list, loss_meter.avg, acc, roc_auc, pr_auc


if __name__ == '__main__':
    #创建保存模型的文件夹
    file_dir = 'checkpoints/res_cfpgan'
    if os.path.exists(file_dir):
        print('true')
        shutil.rmtree(file_dir)
        os.makedirs(file_dir,exist_ok=True)
    else:
        os.makedirs(file_dir)

    # 设置全局参数
    model_lr = 1e-4
    BATCH_SIZE = 16
    EPOCHS = 264
    DEVICE = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    #DEVICE = torch.device("mps")#torch.device("cpu"")
    use_amp = False  # 是否使用混合精度
    use_dp=False #是否开启dp方式的多卡训练
    classes = 2
    resume = False
    CLIP_GRAD = 5.0
    model_path = 'best.pth'
    Best_ACC = 0 #记录最高得分
    use_ema=False
    ema_epoch=32
    # 数据预处理7
    # 数据预处理（增加数据增强）
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=(5,5),sigma=(0.1, 3.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.68023187, 0.7164117, 0.8486238], std= [0.08319076, 0.08293499, 0.06208893])

    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.68023187, 0.7164117, 0.8486238], std= [0.08319076, 0.08293499, 0.06208893])
    ])
    mixup_fn = Mixup(
        mixup_alpha=0., cutmix_alpha=1.0, cutmix_minmax=None,
        prob=1.0, switch_prob=0., mode='batch',
        label_smoothing=0.1, num_classes=classes)
    # 读取数据

    dataset_train = datasets.ImageFolder("/home/itaer2/zxy/shixi/project_1/Baseline model validation for VF progression prediction/Data_eye_CFPs3/train", transform=transform)
    dataset_test = datasets.ImageFolder("/home/itaer2/zxy/shixi/project_1/Baseline model validation for VF progression prediction/Data_eye_CFPs3/val", transform=transform_test)
    with open('class.txt', 'w') as file:
        file.write(str(dataset_train.class_to_idx))
    with open('class.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(dataset_train.class_to_idx))
    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    # 实例化模型并且移动到GPU
    criterion_train = torch.nn.CrossEntropyLoss()
    criterion_val = torch.nn.CrossEntropyLoss()
    #设置模型
    model_ft = timm.create_model('resnet50', pretrained=True, num_classes=2,pretrained_cfg_overlay=dict(file='/home/itaer2/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'))
    print(model_ft)
    if resume:
        model_ft = torch.load(model_path)
    model_ft.to(DEVICE)

    # 选择 Adam 优化器，并加入 L2 正则化（权重衰减）
    weight_decay = 1e-4  # L2 正则化的权重，可调整（通常在 1e-5 到 1e-3 之间）
    optimizer = optim.Adam(model_ft.parameters(), lr=model_lr,weight_decay=weight_decay)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-6)
    if use_amp:
        scaler = torch.amp.GradScaler()
    # if torch.cuda.device_count() > 1 and use_dp:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model_ft = torch.nn.DataParallel(model_ft)
    if use_ema:
        ema = EMA(model_ft, 0.999)
        ema.register()
    # 训练与验证
    is_set_lr = False
    log_dir = {}
    roc_auc_list = []
    pr_auc_list = []
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []
    for epoch in range(1, EPOCHS + 1):
        epoch_list.append(epoch)
        train_loss, train_acc = train(model_ft, DEVICE, train_loader, optimizer, epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        log_dir['train_acc'] = train_acc_list
        log_dir['train_loss'] = train_loss_list
        val_list, pred_list, val_loss, val_acc,roc_auc, pr_auc = val(model_ft, DEVICE, test_loader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        roc_auc_list.append(roc_auc)
        pr_auc_list.append(pr_auc)

        log_dir['val_acc'] = val_acc_list
        log_dir['val_loss'] = val_loss_list
        log_dir['best_acc'] = Best_ACC
        log_dir['roc_auc'] = roc_auc_list
        log_dir['pr_auc'] = pr_auc_list
        with open(file_dir + '/result.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(log_dir))
        print(classification_report(val_list, pred_list, target_names=dataset_train.class_to_idx))
        if epoch < 600:
            cosine_schedule.step()
        else:
            if not is_set_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 1e-6
                    is_set_lr = True
        fig = plt.figure(1)
        plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
        # 显示图例
        plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
        plt.legend(["Train Loss", "Val Loss"], loc="upper right")
        plt.xlabel(u'epoch')
        plt.ylabel(u'loss')
        plt.title('Model Loss ')
        plt.savefig(file_dir + "/loss.png")
        plt.close(1)

        fig2 = plt.figure(2)
        plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
        plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
        plt.legend(["Train Acc", "Val Acc"], loc="lower right")
        plt.title("Model Acc")
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.savefig(file_dir + "/acc.png")
        plt.close(2)

        # 添加 auc 曲线绘图：
        fig3 = plt.figure(3)
        plt.plot(epoch_list, roc_auc_list, 'g-', label=u'ROC-AUC')
        plt.plot(epoch_list, pr_auc_list, 'm-', label=u'PR-AUC')
        plt.xlabel('epoch')
        plt.ylabel('AUC')
        plt.title('Model AUC Curve')
        plt.legend()
        plt.savefig(file_dir + "/auc.png")
        plt.close(3)
