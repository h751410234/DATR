import torch
from torch import nn
import torch.nn.functional as F

def decompose_features(srcs,masks,poss):

    B, _, _, _ = srcs[0].shape
    #source
    srcs_source = []
    masks_source = []
    poss_source = []

    #target
    srcs_target = []
    masks_target = []
    poss_target = []


    for i in range(len(srcs)):
        # source
        srcs_source.append(srcs[i][:B//2,:,:,:])
        masks_source.append(masks[i][:B//2,:,:])
        poss_source.append(poss[i][:B//2,:,:,:])

        srcs_target.append(srcs[i][B // 2:, :, :, :])
        masks_target.append(masks[i][B // 2:, :, :])
        poss_target.append(poss[i][B // 2:, :, :, :])



    return srcs_source,masks_source,poss_source,srcs,masks,poss,srcs_target,masks_target,poss_target

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


class DA_MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


def get_prototype_class_wise(object_query_last_layer,outputs_class,num_classes,global_proto = None,global_amount = None):
    # =====提取原型特征==========
    #(1)获取维度
    B, N, C = object_query_last_layer.shape

    #(2)将预测结果变为label索引
    prob = outputs_class.sigmoid()  # [B,N,num_classes]
    predicted_labels = torch.argmax(prob, dim=2)  # 得到label值 [B,N,1]
    # (3)合并Bs，N维度
    outputs_target = object_query_last_layer.reshape(B * N, 256)  # [BS*N,256]
    outputs_class = predicted_labels.reshape(B * N, 1)  # [BS*N,1]

    # #(4)制备onehot 类别 掩膜图
    onehot = torch.zeros(B * N, num_classes).to(outputs_target.device)  # [BS*N,num_classes]
    onehot.scatter_(dim=1, index=outputs_class, value=1)  # [BS*N,num_classes]
    NxCxA_onehot = onehot.unsqueeze(2).expand(B * N, num_classes, 256)  # [BS*N,num_classes,256]
    #(5)得到存在的类别编号
    class_map = onehot.sum(0)  #[num_classes]
    vaild_class_map = torch.where(class_map != 0, torch.tensor(1), class_map)
    # (6)扩充object维度,并根据类别 mask得到有效值
    outputs_target = outputs_target.view(B * N, 1, 256).expand(B * N, num_classes, 256)
    outputs_target_by_sort = outputs_target.mul(NxCxA_onehot)  # [B*N, self.num_classes, 256])

    # (7)得到Prototypes，暂时不考虑和之前的做合并
    # 对B*N维度求平均
    Amount_CXA = NxCxA_onehot.sum(0)
    Amount_CXA[Amount_CXA == 0] = 1
    prototype_class_wise = outputs_target_by_sort.sum(0) / Amount_CXA  # [self.num_classes, 256]

    # (8)计算权重、使用prototype_class_wise更新全局原型
    sum_weight = class_map.view(num_classes,1).expand(num_classes, 256) #[self.num_classes, 256],用于记录每个类别有效数据的总数量
    weight = sum_weight.div(
        sum_weight + global_amount.view(num_classes, 1).expand(num_classes, 256)
    )  #[self.num_classes, 1]
    weight = weight
    weight[sum_weight == 0] = 0
    global_proto = (global_proto.mul(1 - weight) + prototype_class_wise.mul(weight)).detach()    #[self.num_classes, 256]
    global_amount = global_amount + class_map
    return prototype_class_wise,vaild_class_map,global_proto,global_amount,onehot



