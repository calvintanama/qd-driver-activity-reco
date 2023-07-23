'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import yaml
from collections import OrderedDict


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetLogit(nn.Module):
    def __init__(self, config_file, architecture_config):
    #num_classes=600, sample_size=224, width_mult=1.):
        super(MobileNetLogit, self).__init__()

        num_classes = config_file["train"]["num_classes"]
        width_mult = architecture_config["width_mult"]
        dropout = architecture_config["dropout_prob"]

        input_channel = 32
        last_channel = 1024
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult)
        cfg = [
        # c, n, s
        [64,   1, (2,2,2)],
        [128,  2, (2,2,2)],
        [256,  2, (2,2,2)],
        [512,  6, (2,2,2)],
        [1024, 2, (1,1,1)],
        ]

        self.features = [conv_bn(3, input_channel, (1,2,2))]
        # building inverted residual blocks
        for c, n, s in cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(Block(input_channel, output_channel, stride))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channel, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, config_file, architecture_config):
    #def __init__(self, num_classes=600, sample_size=224, width_mult=1.):
        super(MobileNet, self).__init__()

        num_classes = config_file["train"]["num_classes"]
        width_mult = architecture_config["width_mult"]
        dropout = architecture_config["dropout_prob"]
        #dropout = 0.2

        input_channel = 32
        last_channel = 1024
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult)
        cfg = [
        # c, n, s
        [64,   1, (2,2,2)],
        [128,  2, (2,2,2)],
        [256,  2, (2,2,2)],
        [512,  6, (2,2,2)],
        [1024, 2, (1,1,1)],
        ]

        self.features = [conv_bn(3, input_channel, (1,2,2))]
        # building inverted residual blocks
        for c, n, s in cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(Block(input_channel, output_channel, stride))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), #0.2 originally
            #nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.softmax(x)

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")
    

def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MobileNet(**kwargs)
    return model



if __name__ == '__main__':
    with open("/home/ctanama/framework/config2/mobilenetdrivepre.yaml", "r") as ymlfile:
        #config = yaml.safe_load(ymlfile)
        config = yaml.load(ymlfile, Loader=yaml.Loader)
    #model = get_model(num_classes=600, sample_size = 112, width_mult=1.)
    #model = get_model(config, config['architecture'])
    config['train']["num_classes"] = config['pretraining']["model_num_classes"]
    model = MobileNet(config, config['architecture'])
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    #print(model)
    checkpoint = torch.load("/home/ctanama/pretrained/Pretrained-Models/kinetics_mobilenet_0.5x_RGB_16_best.pth")
    print(len(checkpoint['state_dict']))
    model.load_state_dict(checkpoint['state_dict'])

    print(sum(p.numel() for p in model.parameters()))
    new_dict = OrderedDict()

    for k, v in checkpoint['state_dict'].items():
        new_dict[k[7:]] = v
    #print([k for k,v in new_dict.items()])
    
    same = True

    for k,v in checkpoint['state_dict'].items():
        if not torch.equal(checkpoint['state_dict'][k], new_dict[k[7:]]):
            same = False

    print(same)

    checkpoint = torch.load("/home/ctanama/framework/model/StudentTeacherI3DMobileNetDrivePretraining/exp12/best_model.pth")
    
    new_dict = OrderedDict()

    for k, v in checkpoint['student_model_state_dict'].items():
        newKey = 'module.' + k
        new_dict[newKey] = v

    tens = torch.tensor([[1.,2.], [3.,4.]])
    print(tens)
    tens = tens.view(-1)
    print(tens)
    fill_values = tens[-1]
    print(fill_values)
    total_length = tens.numel()
    print(total_length)
    bucket_size=5
    multiple, rest = divmod(total_length, bucket_size)
    print(multiple, rest)
    if multiple != 0 and rest != 0:
        values_to_add = torch.ones(bucket_size - rest) * fill_values
        tens = torch.cat([tens, values_to_add])
    if multiple == 0:
        tens = tens.view(1,total_length)
    else:
        tens = tens.view(-1, bucket_size)
    print(tens)
    
    config['train']['num_classes'] = 34

    model = MobileNet(config, config['architecture'])
    
    model.load_state_dict(checkpoint['student_model_state_dict'])
    
    #s = set()
    s = torch.Tensor(np.array([]))

    for name, param in model.named_parameters():
        if param.requires_grad:
            s = torch.cat((s, torch.flatten(param)))
            #for data in torch.flatten(param):
                #print(type(data))
                #s.add(data)
                #torch.cat((s,data))

    print("4 bits")
    uniques = torch.unique(s)
    print(uniques.size())

    checkpoint = torch.load("/home/ctanama/framework/model/StudentTeacherI3DMobileNetDrivePretraining/exp13/best_model.pth")

    model.load_state_dict(checkpoint['student_model_state_dict'])

    #s1 = set()
    s1 = torch.Tensor(np.array([]))

    for name, param in model.named_parameters():
        if param.requires_grad:
            s1 = torch.cat((s1, torch.flatten(param)))
            #for data in torch.flatten(param):
                #s1.add(data)

    print("8 bits")
    uniques1 = torch.unique(s1)
    print(uniques1.size())

    checkpoint = torch.load("/home/ctanama/framework/model/StudentTeacherI3DMobileNetDrivePretraining/exp11/best_model.pth")

    model.load_state_dict(checkpoint['student_model_state_dict'])

    s2 = torch.Tensor(np.array([]))

    for name, param in model.named_parameters():
        if param.requires_grad:
            s2 = torch.cat((s2, torch.flatten(param)))
            #for data in torch.flatten(param):
                #s1.add(data)

    print("2 bits")
    uniques2 = torch.unique(s2)
    print(uniques2.size())
    #print(len(s1))
    #input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    #output = model(input_var)
    #print(output.shape)
