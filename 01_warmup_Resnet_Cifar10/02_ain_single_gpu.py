import paddle
import paddle.nn as nn
from resnet18 import ResNet18
from dataset import get_dataset
from dataset import get_dataloader
from utils import AverageMeter
from paddle import optimizer as p_optim
import warnings
warnings.filterwarnings("ignore")

# target: 93%+

def train_one_epoch(model, dataloader, criterion, optimizer, epoch, total_epoch, report_freq=5):
    '''
    训练一个epoch
    '''
    print('=='*20)
    print(f'----- Training Epoch [{epoch}/{total_epoch}]:')
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.train()
    for batch_idx, data in enumerate(dataloader):
        image = data[0]
        label = data[1]

        out = model(image)
        loss = criterion(out, label)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        pred = nn.functional.softmax(out, axis=1)   #计算acc
        acc1 = paddle.metric.accuracy(pred, label.unsqueeze(-1))

        batch_size = image.shape[0]
        loss_meter.update(loss.cpu().numpy()[0], batch_size)
        acc_meter.update(acc1.cpu().numpy()[0], batch_size)

        if batch_idx > 0 and batch_idx % report_freq == 0:
            print(f'---------- Batch[{batch_idx}/{len(dataloader)}], Loss: {loss_meter.avg:.5}, Acc@1: {acc_meter.avg:.4}')
        elif batch_idx > 0 and batch_idx % (max(report_freq//10, 1)) == 0:
            print(f'---------- Batch[{batch_idx}/{len(dataloader)}], Loss: {loss_meter.avg:.5}, Acc@1: {acc_meter.avg:.4}', end='\r')


def validate(model, dataloader, criterion, report_freq=10):
    '''
    用全部测试验证
    '''
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.eval()
    for batch_idx, data in enumerate(dataloader):
        image = data[0]
        label = data[1]

        out = model(image)
        loss = criterion(out, label)

        pred = paddle.nn.functional.softmax(out, axis=1)
        acc1 = paddle.metric.accuracy(pred, label.unsqueeze(-1))
        batch_size = image.shape[0]
        loss_meter.update(loss.cpu().numpy()[0], batch_size)
        acc_meter.update(acc1.cpu().numpy()[0], batch_size)

        '''if batch_idx > 0 and batch_idx % report_freq == 0:
             print(f'---------- Batch [{batch_idx}/{len(dataloader)}], Loss: {loss_meter.avg:.5}, Acc@1: {acc_meter.avg:.4}')'''

    print(f'\n----- Validation: Loss_val: {loss_meter.avg:.5}, Acc_val: {acc_meter.avg:.4}')
    return acc_meter.avg


def main():
    # 设定参数
    total_epoch = 200
    batch_size = 128
    #batch_size = 256

    # 获取模型、数据、损失函数、余弦退火器、Momentum优化器
    model = ResNet18()
    train_dataset = get_dataset(mode='train')
    train_dataloader = get_dataloader(train_dataset, batch_size, mode='train')
    val_dataset = get_dataset(mode='test')
    val_dataloader = get_dataloader(val_dataset, batch_size, mode='test')
    criterion = nn.CrossEntropyLoss()
    # (0.01, total_epoch)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(0.02, total_epoch)
    optimizer = p_optim.Momentum(learning_rate=scheduler,
                                parameters=model.parameters(),
                                momentum=0.9,
                                weight_decay=0.001)
    '''
    #scheduler = paddle.optimizer.lr.CosineAnnealingDecay(0.05, total_epoch)
    optimizer = paddle.optimizer.Momentum(learning_rate=scheduler,
                                          parameters=model.parameters(),
                                          momentum=0.9,
                                          weight_decay=5e-4)
    optimizer = p_optim.AdamW(learning_rate=0.2,
                                        beta1 = 0.9,
                                        beta2 = 0.999,
                                        parameters=model.parameters(),
                                        weight_decay=0.01)  
                                          '''
    # 权重加载
    eval_mode = False
    if eval_mode:
        state_dict = paddle.load('./resnet18_ep200.pdparams')
        model.set_state_dict(state_dict)
        validate(model, val_dataloader, criterion)
        return
    
    #训练
    save_freq = 20
    test_freq = 1
    for epoch in range(1, total_epoch+1):
        train_one_epoch(model, train_dataloader, criterion, optimizer, epoch, total_epoch, report_freq=(10000 // batch_size + 1))
        scheduler.step()

        if epoch % test_freq == 0 or epoch == total_epoch:
            val_acc = validate(model, val_dataloader, criterion)
            if val_acc > 0.93:
                print('Optimization goal has been achieved!')
                paddle.save(model.state_dict(), f'./resnet18_ep{epoch}.pdparams')#Resnet18_2_ep
                paddle.save(optimizer.state_dict(), f'./resnet18_ep{epoch}.pdopts')
                break

        if epoch % save_freq == 0 or epoch == total_epoch:
            paddle.save(model.state_dict(), f'./resnet18_ep{epoch}.pdparams')#Resnet18_2_ep
            paddle.save(optimizer.state_dict(), f'./resnet18_ep{epoch}.pdopts')

if __name__ == "__main__":
    main()
