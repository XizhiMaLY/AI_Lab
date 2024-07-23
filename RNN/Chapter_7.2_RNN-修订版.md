```python
# 导入程序所需要的包

# PyTorch需要的包
import torch
torch.autograd.set_detect_anomaly(True)
import torch.utils.data as DataSet
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import torch.nn.functional as F
# from torchsummary import summary  # 需要预先下载，在终端输入 pip install torchsummary

# 计算需要的包
import string
import numpy as np
import time
```


```python
# 读入并展示数据
f = open('./poems_clean.txt', "r", encoding='utf-8')
poems = []
for line in f.readlines():
    title, poem = line.split(':')
    poem = poem.replace(' ', '')
    poem = poem.replace('\n', '')
    if len(poem) > 0 :
        poems.append(list(poem))
    
print(poems[0][:])
```

    ['寒', '随', '穷', '律', '变', '春', '逐', '鸟', '声', '开', '初', '风', '飘', '带', '柳', '晚', '雪', '间', '花', '梅', '碧', '林', '青', '旧', '竹', '绿', '沼', '翠', '新', '苔', '芝', '田', '初', '雁', '去', '绮', '树', '巧', '莺', '来']



```python
# 创建字符编码字典
word2idx = {}
i = 1
for poem in poems:
    for word in poem:
        if word2idx.get(word) == None:
            word2idx[word] = i
            i += 1
            
list(word2idx.items())[:10]
```




    [('寒', 1),
     ('随', 2),
     ('穷', 3),
     ('律', 4),
     ('变', 5),
     ('春', 6),
     ('逐', 7),
     ('鸟', 8),
     ('声', 9),
     ('开', 10)]




```python
len(word2idx)
```




    5545




```python
# 对诗歌进行编码，从原始数据到矩阵
poems_digit = []
for poem in poems:
    poem_digit = []
    for word in poem:
        poem_digit.append(word2idx[word])
    poems_digit.append(poem_digit)
    
print("原始诗歌")
print(poems[3829])
print("\n 编码后的结果")
print(poems_digit[3829][:])
```

    原始诗歌
    ['春', '眠', '不', '觉', '晓', '处', '处', '闻', '啼', '鸟', '夜', '来', '风', '雨', '声', '花', '落', '知', '多', '少']
    
     编码后的结果
    [6, 2420, 57, 2468, 451, 198, 198, 747, 376, 8, 228, 39, 12, 270, 9, 19, 319, 67, 510, 1941]



```python
# 拆分X、Y变量并处理长短不一问题
# 设置诗歌最大长度为50个字符
maxlen = 50
X = []
Y = []
for poem_digit in poems_digit:
    y=poem_digit[1:]+[0]*(maxlen - len(poem_digit)) # 此处修改y 
    Y.append(y)
    # 将最后一个字符之前的部分作为X，并补齐字符
    x = poem_digit[:-1] + [0]*(maxlen - len(poem_digit))
    X.append(x)
    
print("原始诗歌")
print(poems[3829])
print("变量X")
print(X[3829])
print("变量Y")
print(Y[3829])
```

    原始诗歌
    ['春', '眠', '不', '觉', '晓', '处', '处', '闻', '啼', '鸟', '夜', '来', '风', '雨', '声', '花', '落', '知', '多', '少']
    变量X
    [6, 2420, 57, 2468, 451, 198, 198, 747, 376, 8, 228, 39, 12, 270, 9, 19, 319, 67, 510, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    变量Y
    [2420, 57, 2468, 451, 198, 198, 747, 376, 8, 228, 39, 12, 270, 9, 19, 319, 67, 510, 1941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



```python
# 划分训练集和测试集
# 将所有数据的顺序打乱重排
idx = np.random.permutation(range(len(X)))
X = [X[i] for i in idx]
Y = [Y[i] for i in idx]

# 切分出1/5的数据放入校验集    
validX = X[:len(X) // 5]
trainX = X[len(X) // 5:]
validY = Y[:len(Y) // 5]
trainY = Y[len(Y) // 5:]
```


```python
'''
将数据转化为dataset，并用dataloader来加载数据。dataloader是PyTorch开发采用的一套管理数据的方法。通常数据储存在dataset中，而对数据的调用则由
dataloader完成。同时，在预处理时，系统已经自动将数据打包成batch，每次调用都取出一批。从dataloader中输出的每一个元素都是一个(x,y)元组，x为输
入张量，y为标签。x和y的第一个维度都是batch_size大小。
'''

# 一批包含64个数据记录。这个数字越大，系统训练时，每一个周期要处理的数据就越多、处理就越快，但总数据量会减少。
batch_size = 64
# 形成训练集
train_ds = DataSet.TensorDataset(torch.IntTensor(np.array(trainX, dtype=int)), torch.IntTensor(np.array(trainY, dtype=int)))
# 形成数据加载器
train_loader = DataSet.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

# 校验数据
valid_ds = DataSet.TensorDataset(torch.IntTensor(np.array(validX, dtype=int)), torch.IntTensor(np.array(validY, dtype=int)))
valid_loader = DataSet.DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4)
```


```python
class RNN_py(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, batch_first=True):
        super(RNN_py, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Initialize parameters that can be updated
        self.weight_ih = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, embedding_size)) for _ in range(num_layers)
        ])
        self.bias_ih = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size)) for _ in range(num_layers)
        ])
        self.weight_hh = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(num_layers)
        ])
        self.bias_hh = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size)) for _ in range(num_layers)
        ])
    
    def forward(self, x, h_0=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h_t_minus_1 = h_0
        h_t = h_0
        output = []
        for t in range(seq_len):
            for layer in range(self.num_layers):
                h_t[layer] = torch.tanh(
                    x[t] @ self.weight_ih[layer].T
                    + self.bias_ih[layer]
                    + h_t_minus_1[layer] @ self.weight_hh[layer].T
                    + self.bias_hh[layer]
                )
            output.append(h_t[-1])
            h_t_minus_1 = h_t
        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_t
```


```python
'''
实现一个简单的RNN，其构架主要包含3层：输入层，一层隐含层和输出层
先是embedding层，将词表转化成embedding
后面是hidden size
最后接一个输出为词表维度的线性层，接crossentropy求loss
'''

class SimpleRNN(nn.Module):
    def __init__(self, output_size, word_num, embedding_size, hidden_size, num_layers=1):
        # 定义
        super(SimpleRNN, self).__init__()
        
        # 一个embedding层
        self.embedding = nn.Embedding(word_num, embedding_size)
        
        # PyTorch的RNN层，batch_first标识可以让输入的张量的第一个维度表示batch指标
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True)
        self.rnn_py = RNN_py(embedding_size, hidden_size, num_layers, batch_first=True)
        
        # 输出的全连接层
        self.fc = nn.Linear(hidden_size, output_size)
                
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def forward(self, x, hidden):
        # 运算过程
        # 先进行embedding层的计算
        x = self.embedding(x)
        # 从输入到隐含层的计算
        # x的尺寸为：batch_size，num_step，hidden_size
        output, hidden = self.rnn_py(x, hidden)
        # output的尺寸为：batch_size，maxlen-1, hidden_size
        # 最后一层全连接网络 此处返回每个时间步的数值
        output = self.fc(output)
        output = output.view(-1,output.shape[-1])#为便于后续处理，此处进行展平
        # output的尺寸为：batch_size*(maxlen-1)，output_size
        return output, hidden
    
    def initHidden(self, batch_size):
        # 对隐含单元初始化
        # 尺寸是layer_size，batch_size，hidden_size
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
```


```python
# 获取文本数据集中包含的字符数量
vocab_size = len(word2idx.keys()) + 1

# 给定超参数
lr = 1e-3
epochs = 50

# 生成一个简单的RNN，输入size为49（50-1），输出size为vocab_size（字符总数）
rnn = SimpleRNN(output_size=vocab_size, word_num=vocab_size, embedding_size=64, hidden_size=128)
rnn = rnn.cuda()
criterion = torch.nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr) #Adam优化算法
# 查看模型具体信息
print(rnn)
```

    SimpleRNN(
      (embedding): Embedding(5546, 64)
      (rnn): RNN(64, 128, batch_first=True)
      (rnn_py): RNN_py(
        (weight_ih): ParameterList(  (0): Parameter containing: [torch.float32 of size 128x64 (cuda:0)])
        (bias_ih): ParameterList(  (0): Parameter containing: [torch.float32 of size 128 (cuda:0)])
        (weight_hh): ParameterList(  (0): Parameter containing: [torch.float32 of size 128x128 (cuda:0)])
        (bias_hh): ParameterList(  (0): Parameter containing: [torch.float32 of size 128 (cuda:0)])
      )
      (fc): Linear(in_features=128, out_features=5546, bias=True)
    )



```python
'''
计算预测错误率的函数，pre是模型给出的一组预测结果（batch_size行、num_classes列的矩阵），label是正确标签
'''

def accuracy(pre, label):
    # 得到每一行（每一个样本）输出值最大元素的下标
    pre = torch.max(pre.data, 1)[1]
    # 将下标与label比较，计算正确的数量
    rights = pre.eq(label.data).sum()
    # 计算正确预测所占百分比
    acc = rights.data/len(label)
    return acc.float()
```


```python
# 模型验证
def validate(model, val_loader):
    # 在校验集上运行一遍并计算损失和准确率
    val_loss = 0
    val_acc = 0
    model.eval()
    for batch, data in enumerate(val_loader):
        init_hidden = model.initHidden(len(data[0]))
        init_hidden = init_hidden.cuda()
        x, y = Variable(data[0]), Variable(data[1])
        x, y = x.cuda(), y.cuda()
        outputs, hidden = model(x, init_hidden)
        y = y.long()
        y = y.view(y.shape[0]*y.shape[1]) #此处修改：展平，对应x的维度
        loss = criterion(outputs, y)
        val_loss += loss.data.cpu().numpy()  
        val_acc += accuracy(outputs, y)
    val_loss /= len(val_loader)  # 计算平均损失
    val_acc /= len(val_loader)  # 计算平均准确率
    return val_loss, val_acc
```


```python
# 打印训练结果
def print_log(epoch, train_time, train_loss, train_acc, val_loss, val_acc, epochs=10):
    print(f"Epoch [{epoch}/{epochs}], time: {train_time:.2f}s, loss: {train_loss:.4f}, acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
```


```python
# 定义主函数：模型训练
def train(model,optimizer, train_loader, val_loader, epochs=1):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        start = time.time()  # 记录本epoch开始时间
        for batch, data in enumerate(train_loader):
            # batch为数字，表示已经进行了几个batch
            # data为一个二元组，存储了一个样本的输入和标签
            model.train() # 标志当前RNN处于训练阶段
            init_hidden = model.initHidden(len(data[0])) # 初始化隐含层单元
            init_hidden = init_hidden.cuda()
            optimizer.zero_grad()
            x, y = Variable(data[0]), Variable(data[1]) # 从数据中提取输入和输出对
            x, y = x.cuda(), y.cuda()
            outputs, hidden = model(x, init_hidden) # 输入RNN，产生输出
            y = y.long()
            y = y.view(y.shape[0]*y.shape[1]) #此处修改：展平，对应x的维度
            loss = criterion(outputs, y) # 带入损失函数并产生loss
            train_loss += loss.data.cpu().numpy()  # 记录loss
            train_acc += accuracy(outputs, y) # 记录acc
            loss.backward() # 反向传播
            optimizer.step() # 梯度更新
        
        end = time.time()  # 记录本epoch结束时间
        train_time = end - start  # 计算本epoch的训练耗时 
        train_loss /= len(train_loader)  # 计算平均损失
        train_acc /= len(train_loader)  # 计算平均准确率             
        val_loss, val_acc = validate(model, val_loader)  # 计算测试集上的损失函数和准确率
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc) 
        print_log(epoch + 1, train_time, train_loss, train_acc, val_loss, val_acc, epochs=epochs)  # 打印训练结果
    return train_losses, train_accs, val_losses, val_accs
```


```python
# 模型训练
history = train(rnn, optimizer, train_loader, valid_loader, epochs=epochs)  # 实施训练
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[18], line 2
          1 # 模型训练
    ----> 2 history = train(rnn, optimizer, train_loader, valid_loader, epochs=epochs)  # 实施训练


    Cell In[17], line 12, in train(model, optimizer, train_loader, val_loader, epochs)
         10 train_acc = 0
         11 start = time.time()  # 记录本epoch开始时间
    ---> 12 for batch, data in enumerate(train_loader):
         13     # batch为数字，表示已经进行了几个batch
         14     # data为一个二元组，存储了一个样本的输入和标签
         15     model.train() # 标志当前RNN处于训练阶段
         16     init_hidden = model.initHidden(len(data[0])) # 初始化隐含层单元


    File ~/miniconda3/envs/cops3/lib/python3.10/site-packages/torch/utils/data/dataloader.py:631, in _BaseDataLoaderIter.__next__(self)
        628 if self._sampler_iter is None:
        629     # TODO(https://github.com/pytorch/pytorch/issues/76750)
        630     self._reset()  # type: ignore[call-arg]
    --> 631 data = self._next_data()
        632 self._num_yielded += 1
        633 if self._dataset_kind == _DatasetKind.Iterable and \
        634         self._IterableDataset_len_called is not None and \
        635         self._num_yielded > self._IterableDataset_len_called:


    File ~/miniconda3/envs/cops3/lib/python3.10/site-packages/torch/utils/data/dataloader.py:1329, in _MultiProcessingDataLoaderIter._next_data(self)
       1326     return self._process_data(data)
       1328 assert not self._shutdown and self._tasks_outstanding > 0
    -> 1329 idx, data = self._get_data()
       1330 self._tasks_outstanding -= 1
       1331 if self._dataset_kind == _DatasetKind.Iterable:
       1332     # Check for _IterableDatasetStopIteration


    File ~/miniconda3/envs/cops3/lib/python3.10/site-packages/torch/utils/data/dataloader.py:1295, in _MultiProcessingDataLoaderIter._get_data(self)
       1291     # In this case, `self._data_queue` is a `queue.Queue`,. But we don't
       1292     # need to call `.task_done()` because we don't use `.join()`.
       1293 else:
       1294     while True:
    -> 1295         success, data = self._try_get_data()
       1296         if success:
       1297             return data


    File ~/miniconda3/envs/cops3/lib/python3.10/site-packages/torch/utils/data/dataloader.py:1133, in _MultiProcessingDataLoaderIter._try_get_data(self, timeout)
       1120 def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
       1121     # Tries to fetch data from `self._data_queue` once for a given timeout.
       1122     # This can also be used as inner loop of fetching without timeout, with
       (...)
       1130     # Returns a 2-tuple:
       1131     #   (bool: whether successfully get data, any: data if successful else None)
       1132     try:
    -> 1133         data = self._data_queue.get(timeout=timeout)
       1134         return (True, data)
       1135     except Exception as e:
       1136         # At timeout and error, we manually check whether any worker has
       1137         # failed. Note that this is the only mechanism for Windows to detect
       1138         # worker failures.


    File ~/miniconda3/envs/cops3/lib/python3.10/multiprocessing/queues.py:113, in Queue.get(self, block, timeout)
        111 if block:
        112     timeout = deadline - time.monotonic()
    --> 113     if not self._poll(timeout):
        114         raise Empty
        115 elif not self._poll():


    File ~/miniconda3/envs/cops3/lib/python3.10/multiprocessing/connection.py:257, in _ConnectionBase.poll(self, timeout)
        255 self._check_closed()
        256 self._check_readable()
    --> 257 return self._poll(timeout)


    File ~/miniconda3/envs/cops3/lib/python3.10/multiprocessing/connection.py:424, in Connection._poll(self, timeout)
        423 def _poll(self, timeout):
    --> 424     r = wait([self], timeout)
        425     return bool(r)


    File ~/miniconda3/envs/cops3/lib/python3.10/multiprocessing/connection.py:931, in wait(object_list, timeout)
        928     deadline = time.monotonic() + timeout
        930 while True:
    --> 931     ready = selector.select(timeout)
        932     if ready:
        933         return [key.fileobj for (key, events) in ready]


    File ~/miniconda3/envs/cops3/lib/python3.10/selectors.py:416, in _PollLikeSelector.select(self, timeout)
        414 ready = []
        415 try:
    --> 416     fd_event_list = self._selector.poll(timeout)
        417 except InterruptedError:
        418     return ready


    KeyboardInterrupt: 



```python
# 使用RNN写藏头诗
# 初始化藏头诗字符串
poem_incomplete = '深****度****学****习****'
poem_index = [] # 用于记录诗歌创作过程中字符和整数的对应关系
poem_text = '' # 记录诗歌的创作过程，循环结束后应是一首完整的诗

init_hidden = rnn.initHidden(1)
init_hidden = init_hidden.cuda()
for i in range(len(poem_incomplete)):
    # 对poem_incomplete的每个字符做循环
    current_word = poem_incomplete[i]
    if current_word != '*':
        # 若当前的字符不是"*"，使用word2idx字典将其变为一个整数
        index = word2idx[current_word]
    else:
        # 若当前的字符是"*"，需要用RNN模型对其进行预测
        x = poem_index + [0]*(maxlen -1 - len(poem_index)) # 将当前字符与之前的字符拼接形成新的输入序列
        x = torch.IntTensor(np.array([x], dtype=int))
        x = Variable(x)
        x = x.cuda()
        pre, hidden = rnn(x, init_hidden) 
        init_hidden = hidden.data # 更新隐藏状态
        crt_pre = pre[i-1].cpu() # 获取第i个时间步的输出，对应位置i-1
        index = torch.argmax(crt_pre) # 提取最大概率的字符所在的位置，记录其编号
        current_word = [k for k, v in word2idx.items() if v == index][0] # 提取上述编号所对应的字符
    poem_index.append(index) 
    poem_text = poem_text + current_word # 将current_word加到poem_text中
```


```python
print(poem_text[0:5])
print(poem_text[5:10])
print(poem_text[10:15])
print(poem_text[15:20])
```

    深山下石溪
    度月明月明
    学得意何处
    习家家女泪



```python

```
