# 李宏毅2021/2022机器学习作业上手集
## HW1
### 框架
#### 文件处理
通过在extract_file()函数中，使用un_zip(file_name)将.zip文件解压。其中用for循环对文件夹进行后缀是.zip的文件进行查找，`if file_name.endswith('.zip')`，通过`extract_dir = file_name.rsplit('.',1)[0] + "_files"`提取zip文件的名字，`os.mkdir(‘文件夹名’,exist_ok = True)`创立文件夹， `zip_file.extractall(extract_dir)` 将 zip_file 中的所有文件和目录提取到 extract_dir 目录中。
#### 数据集
数据集名字是`class COVID19Dataset` ，里面规定了mode ，X，Y ，通过if和else可以将train、test的
train_x，train_y，或者test_x，test_y赋值给X和Y。其中`train_data_raw = pd.read_csv(os.path.join(file_path,'covid.train.csv')).drop(columns=['id']).values`函数与` self.X = torch.FloatTensor(self.X)`重要。并注意应该对X进行`normalize`，同时不要忘了`self.dim = self.X.shape[1]`。

#### 加载数据集
首先应该注意`Dataloader()`函数返回的是`dataloader`类型的数据。
```python
def prep_dataloader(path,mode,batch_size,n_jobs=0,target_only=False):
    dataset = COVID19Dataset(path,mode=mode,target_only=target_only)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'),# 训练模式下打乱数据
        drop_last=False,#不丢弃最后一个不完整的batch
        num_workers=n_jobs, ## 使用的线程数
        pin_memory=True) # Construct dataloader
    return dataloader
```
#### 神经网络
通过`import torch.nn as nn`，`class NeuralNet(nn.Module)` 初始化神经网络需要定义`forward(self,x)`和`cal_loss(self,pred,target)`函数。
#### Setup Hyper-parameters
```python
config = {
    'n_epochs':3000,
    'batch_size':30,
    'optimizer':'SGD',#
    'optim_hparas':{
        'lr' : 1e-4,
        'momentum':0.9
    },
    'early_stop':200
}
```
#### Training 
优化器optimizer，n_epochs ，`loss_record = {'train':[],'dev':[]} 和 loss_record['train'].append(mse_loss.detach().cpu().item()) `记录loss。
```python
for epoch in range(n_epochs):
        model.train()
        for x,y in train_set:
            optimizer.zero_grad()
            x,y = x.to(device) , y.to(device)
            pred = model(x)
            mse_loss = model.cal_loss(pred,y) 
            mse_loss.backward()
            optimizer.step()
```
####  Testing和 Validation 和 load data
这两个函数都需要model.eval() ，并且在使用时开启`with torch.no_grad()`。
`model = NeuralNet(tr_set.dataset.dim).to(device)`加载model。
#### 保存csv文件
关键的函数有两个：`df = pd.DataFrame({'id':range(len(preds)),'tested_positive':preds[:]})`和`df.to_csv(submission_path, index=False)`

## HW2
等作者有力气的时候在写吧~

