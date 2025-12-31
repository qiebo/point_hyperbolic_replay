# 导入必要的库
import torch  # 导入PyTorch库，用于深度学习
import numpy as np  # 导入NumPy库，用于数值计算
import csv  # 导入CSV库，用于处理CSV文件

# 定义一个函数，用于在一维张量前后填充零，并添加一个维度
def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # 将一维张量中的0替换为1
    xlen = x.size(0)  # 获取张量的长度
    if xlen < padlen:  # 如果长度小于指定填充长度
        new_x = x.new_zeros([padlen], dtype=x.dtype)  # 创建一个新的全零张量，长度为padlen
        new_x[:xlen] = x  # 将原始张量的值复制到新的张量中
        x = new_x  # 将新的张量赋给x
    return x.unsqueeze(0)  # 在张量的前面添加一个维度，变成二维张量

# 定义一个函数，用于在一维张量前面填充零，并添加一个维度
def pad_1d_unsqueeze2(x, padlen):
    xlen = x.size(0)  # 获取张量的长度
    if xlen < padlen:  # 如果长度小于指定填充长度
        new_x = x.new_zeros([padlen], dtype=x.dtype)  # 创建一个新的全零张量，长度为padlen
        new_x[:xlen] = x  # 将原始张量的值复制到新的张量中
        x = new_x  # 将新的张量赋给x
    return x.unsqueeze(0)  # 在张量的前面添加一个维度，变成二维张量

# 定义一个函数，用于在二维张量前后填充零，并添加一个维度
def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # 将二维张量中的0替换为1
    xlen, xdim = x.size()  # 获取二维张量的行数和列数
    if xlen < padlen:  # 如果行数小于指定填充长度
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)  # 创建一个新的全零张量，形状为(padlen, xdim)
        new_x[:xlen, :] = x  # 将原始张量的值复制到新的张量中
        x = new_x  # 将新的张量赋给x
    return x.unsqueeze(0)  # 在张量的前面添加一个维度，变成三维张量

# 定义一个函数，用于在注意力偏置张量前后填充负无穷，并添加一个维度
def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)  # 获取张量的长度
    if xlen < padlen:  # 如果长度小于指定填充长度
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))  # 创建一个新的全零张量，形状为(padlen, padlen)，并填充为负无穷
        new_x[:xlen, :xlen] = x  # 将原始张量的值复制到新的张量中
        new_x[xlen:, :xlen] = 0  # 将新的张量的后部分设置为0
        x = new_x  # 将新的张量赋给x
    return x.unsqueeze(0)  # 在张量的前面添加一个维度，变成三维张量

# 定义一个函数，用于在边类型张量前面填充零，并添加一个维度
def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)  # 获取张量的长度
    if xlen < padlen:  # 如果长度小于指定填充长度
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)  # 创建一个新的全零张量，形状为(padlen, padlen, x.size(-1))
        new_x[:xlen, :xlen, :] = x  # 将原始张量的值复制到新的张量中
        x = new_x  # 将新的张量赋给x
    return x.unsqueeze(0)  # 在张量的前面添加一个维度，变成四维张量

# 定义一个函数，用于在二维张量前面填充零，并添加一个维度
def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1  # 将二维张量中的0替换为1
    xlen = x.size(0)  # 获取张量的长度
    if xlen < padlen:  # 如果长度小于指定填充长度
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)  # 创建一个新的全零张量，形状为(padlen, padlen)
        new_x[:xlen, :xlen] = x  # 将原始张量的值复制到新的张量中
        x = new_x  # 将新的张量赋给x
    return x.unsqueeze(0)  # 在张量的前面添加一个维度，变成三维张量

# 定义一个函数，用于在三维张量前面填充零，并添加一个维度
def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1  # 将三维张量中的0替换为1
    xlen1, xlen2, xlen3, xlen4 = x.size()  # 获取三维张量的形状
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:  # 如果任何一个维度小于指定填充长度
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)  # 创建一个新的全零张量，形状为(padlen1, padlen2, padlen3, xlen4)
        new_x[:xlen1, :xlen2, :xlen3, :] = x  # 将原始张量的值复制到新的张量中
        x = new_x  # 将新的张量赋给x
    return x.unsqueeze(0)  # 在张量的前面添加一个维度，变成四维张量

# 定义一个计算Forman曲率的函数，这是一个三步计算方法
def Formancurvature_3_step(in_degrees, spatial_poses):
    N = in_degrees.size(0)  # 获取节点数量
    temp = [0 for i in range(N)]  # 创建一个临时列表，用于存储计算结果
    indegree_j = 0  # 初始化节点j的入度
    indegree_k = 0  # 初始化节点k的入度
    for j in range(N):
        for k in range(j, N):
            if spatial_poses[j][k] == 3:  # 如果节点j和节点k之间的空间位置为3
                indegree_j += 1  # 节点j的入度加1
                indegree_k += 1  # 节点k的入度加1
    for j in range(0, N):
        for k in range(j, N):
            if spatial_poses[j][k] == 3:  # 如果节点j和节点k之间的空间位置为3
                w = (4 - indegree_j - indegree_k) * 0.0625  # 计算权重w
                temp[j] += (w / (indegree_j))  # 更新节点j的临时值
                temp[k] += (w / (indegree_k))  # 更新节点k的临时值
    return np.array(temp)  # 返回计算结果的NumPy数组

# 定义一个计算Forman曲率的函数，这是一个两步计算方法
def Formancurvature_2_step(in_degrees, spatial_poses):
    N = in_degrees.size(0)  # 获取节点数量
    temp = [0 for i in range(N)]  # 创建一个临时列表，用于存储计算结果
    indegree_j = 0  # 初始化节点j的入度
    indegree_k = 0  # 初始化节点k的入度
    for j in range(N):
        for k in range(j, N):
            if spatial_poses[j][k] == 2:  # 如果节点j和节点k之间的空间位置为2
                indegree_j += 1  # 节点j的入度加1
                indegree_k += 1  # 节点k的入度加1
    for j in range(0, N):
        for k in range(j, N):
            if spatial_poses[j][k] == 2:  # 如果节点j和节点k之间的空间位置为2
                w = (4 - indegree_j - indegree_k) * 0.50  # 计算权重w
                temp[j] += (w / (indegree_j))  # 更新节点j的临时值
                temp[k] += (w / (indegree_k))  # 更新节点k的临时值
    return np.array(temp)  # 返回计算结果的NumPy数组

# 定义一个计算Forman曲率的函数，根据输入的in_degrees和spatial_poses计算曲率
def Formancurvature(in_degrees, spatial_poses):
    batchsize = len(in_degrees)  # 获取批次大小
    ave_forman_curvatures = ()  # 创建一个空元组，用于存储计算结果
    for i in range(batchsize):
        N = in_degrees[i].size(0)  # 获取节点数量
        temp = [0 for i in range(N)]  # 创建一个临时列表，用于存储计算结果
        for j in range(N):
            for k in range(j, N):
                if spatial_poses[i][j][k] == 1:  # 如果节点j和节点k之间的空间位置为1
                    w = (4 - in_degrees[i][j] - in_degrees[i][k])  # 计算权重w
                    temp[j] += (w / (in_degrees[i][j])).item()  # 更新节点j的临时值
                    temp[k] += (w / (in_degrees[i][k])).item()  # 更新节点k的临时值
        temp = np.array(temp)  # 将临时列表转换为NumPy数组
        # K-hop Curvature
        if N > 34:  # 如果节点数量大于34
            HOP2 = Formancurvature_2_step(in_degrees[i], spatial_poses[i])  # 使用两步计算方法计算Forman曲率
            HOP3 = Formancurvature_3_step(in_degrees[i], spatial_poses[i])  # 使用三步计算方法计算Forman曲率
            temp = (temp + HOP2 + HOP3)  # 更新临时值
        ave_forman_curvatures += (torch.from_numpy(temp),)  # 将计算结果添加到元组中
    return ave_forman_curvatures  # 返回计算结果的元组

# 定义一个计算Forman曲率的函数，这是一个一步计算方法
def Formancurvature_1_step(in_degrees, spatial_poses):
    batchsize = len(in_degrees)  # 获取批次大小
    ave_forman_curvatures = ()  # 创建一个空元组，用于存储计算结果
    for i in range(batchsize):
        N = in_degrees[i].size(0)  # 获取节点数量
        temp = [0 for i in range(N)]  # 创建一个临时列表，用于存储计算结果
        for j in range(N):
            for k in range(j, N):
                if spatial_poses[i][j][k] == 1:  # 如果节点j和节点k之间的空间位置为1
                    w = (4 - in_degrees[i][j] - in_degrees[i][k])  # 计算权重w
                    temp[j] += (w / (in_degrees[i][j])).item()  # 更新节点j的临时值
                    temp[k] += (w / (in_degrees[i][k])).item()  # 更新节点k的临时值
        temp = np.array(temp)  # 将临时列表转换为NumPy数组
        ave_forman_curvatures += (torch.from_numpy(temp),)  # 将计算结果添加到元组中
    return ave_forman_curvatures  # 返回计算结果的元组

# 定义一个计算Ollivier粗曲率的函数
def Ollivier_coarse_curvature(in_degrees, spatial_poses):
    batchsize = len(in_degrees)  # 获取批次大小
    coarse_curvature = ()  # 创建一个空元组，用于存储计算结果
    for i in range(batchsize):
        N = in_degrees[i].size(0)  # 获取节点数量
        edge_curvature = [[0 for col in range(N)] for row in range(N)]  # 创建一个二维列表，用于存储边的曲率
        for j in range(N):
            for k in range(j, N):
                A = []  # 创建一个空列表，用于存储与节点j相邻的节点
                B = []  # 创建一个空列表，用于存储与节点k相邻的节点
                if spatial_poses[i][j][k] == 1:  # 如果节点j和节点k之间的空间位置为1
                    if in_degrees[i][j] == 1 or in_degrees[i][k] == 1:  # 如果节点j或节点k的入度为1
                        edge_curvature[j][k] = 0  # 边的曲率为0
                    else:
                        mx = max(in_degrees[i][j], in_degrees[i][k])  # 计算节点j和节点k的入度的最大值
                        for t in range(N):
                            if spatial_poses[i][j][t] == 1:  # 如果节点j与节点t相邻
                                A.append(t)  # 将节点t添加到列表A中
                            if spatial_poses[i][k][t] == 1:  # 如果节点k与节点t相邻
                                B.append(t)  # 将节点t添加到列表B中
                        for ii in range(len(A)):
                            for jj in range(len(B)):
                                edge_curvature[j][k] += spatial_poses[i][A[ii]][B[jj]]  # 计算边的曲率
                        edge_curvature[j][k] = 1 - (edge_curvature[j][k] / mx)  # 更新边的曲率
        temp = [0 for i in range(N)]  # 创建一个临时列表，用于存储节点的曲率
        for j in range(N):
            for k in range(j, N):
                if spatial_poses[i][j][k] == 1:  # 如果节点j和节点k之间的空间位置为1
                    temp[j] += edge_curvature[j][k]  # 更新节点j的临时值
                    temp[k] += edge_curvature[j][k]  # 更新节点k的临时值
        for k in range(N):
            temp[k] = temp[k] / in_degrees[i][k]  # 计算节点的曲率
        temp = np.array(temp)  # 将临时列表转换为NumPy数组
        coarse_curvature += (torch.from_numpy(temp),)  # 将计算结果添加到元组中
    return coarse_curvature  # 返回计算结果的元组

# 定义一个数据处理函数，用于将输入数据整理成模型训练所需的格式
def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]  # 过滤掉不符合条件的数据项
    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
        )
        for item in items
    ]  # 从数据项中提取需要的字段，并组成元组
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
    ) = zip(*items)  # 将元组中的字段解压缩成多个列表

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")  # 根据条件将注意力偏置中的部分值设置为负无穷
    max_node_num = max(i.size(0) for i in xs)  # 计算节点特征的最大长度
    max_dist = max(i.size(-2) for i in edge_inputs)  # 计算边距离的最大长度
    y = torch.cat(ys)  # 将目标值合并为一个张量
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])  # 对节点特征进行填充和合并
    ave_forman_curvatures_1 = Formancurvature_1_step(in_degrees, spatial_poses)  # 计算Forman曲率
    torch.set_printoptions(threshold=np.inf)  # 设置打印选项，以便打印所有元素

    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )  # 对边的输入进行填充和合并
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )  # 对注意力偏置进行填充和合并
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )  # 对边类型进行填充和合并

    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )  # 对空间位置进行填充和合并
    in_degree = torch.cat(
        [pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]
    )  # 对入度进行填充和合并
    out_degree = torch.cat(
        [pad_1d_unsqueeze(i, max_node_num) for i in out_degrees]
    )  # 对出度进行填充和合并
    ave_forman_curvatures_1 = torch.cat(
        [pad_1d_unsqueeze2(i, max_node_num) for i in ave_forman_curvatures_1]
    )  # 对Forman曲率进行填充和合并

    return (
        idxs,
        attn_bias,
        attn_edge_type,
        spatial_pos,
        in_degree,
        out_degree,
        x,
        edge_input,
        y,
        ave_forman_curvatures_1,
    )  # 返回整理后的数据

# 定义一个类，用于表示图数据的一项
class GraphItem:
    def __init__(
            self,
            idx,
            attn_bias,
            attn_edge_type,
            spatial_pos,
            in_degree,
            out_degree,
            x,
            edge_input,
            y,
    ):
        self.idx = idx  # 图的索引
        self.attn_bias = attn_bias  # 注意力偏置
        self.attn_edge_type = attn_edge_type  # 注意力边类型
        self.spatial_pos = spatial_pos  # 空间位置
        self.in_degree = in_degree  # 入度
        self.out_degree = out_degree  # 出度
        self.x = x  # 节点特征
        self.edge_input = edge_input  # 边的输入
        self.y = y  # 目标值

# 定义一个类，用于加载数据集
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.data = []
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                idx, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, x, edge_input, y = row
                idx = int(idx)
                attn_bias = torch.tensor(eval(attn_bias), dtype=torch.float32)
                attn_edge_type = torch.tensor(eval(attn_edge_type), dtype=torch.float32)
                spatial_pos = torch.tensor(eval(spatial_pos), dtype=torch.float32)
                in_degree = torch.tensor(eval(in_degree), dtype=torch.float32)
                out_degree = torch.tensor(eval(out_degree), dtype=torch.float32)
                x = torch.tensor(eval(x), dtype=torch.float32)
                edge_input = torch.tensor(eval(edge_input), dtype=torch.float32)
                y = torch.tensor(eval(y), dtype=torch.float32)
                item = GraphItem(
                    idx, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, x, edge_input, y
                )
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 加载数据集
train_dataset = MyDataset("train.csv")  # 训练集
valid_dataset = MyDataset("valid.csv")  # 验证集
test_dataset = MyDataset("test.csv")    # 测试集

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True, collate_fn=collator
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=16, shuffle=False, collate_fn=collator
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=16, shuffle=False, collate_fn=collator
)
