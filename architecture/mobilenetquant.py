'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import yaml
from collections import OrderedDict
import os


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
        self.relu = nn.ReLU();

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetLogit(nn.Module):
    def __init__(self, config_file, architecture_config):
    #num_classes=600, sample_size=224, width_mult=1.):
        super(MobileNetLogit, self).__init__()

        num_classes = config_file["train"]["num_classes"]
        width_mult = architecture_config["width_mult"]
        dropout = architecture_config["dropout_prob"]
        n_frame = config_file["data"]["n_frame"]
        frame_size = config_file["data"]["frame_size"]
        kernel = architecture_config['pool']['kernel']
        width = architecture_config['pool']['width']
        height = architecture_config['pool']['height']

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
        self.avgpool3d = nn.AvgPool3d((kernel, width, height))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channel, num_classes),
        )
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool3d(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, config_file, architecture_config):
    #def __init__(self, num_classes=600, sample_size=224, width_mult=1.):
        super(MobileNet, self).__init__()

        num_classes = config_file["train"]["num_classes"]
        width_mult = architecture_config["width_mult"]
        dropout = architecture_config["dropout_prob"]
        n_frame = config_file["data"]["n_frame"]
        frame_size = config_file["data"]["frame_size"]
        kernel = architecture_config['pool']['kernel']
        width = architecture_config['pool']['width']
        height = architecture_config['pool']['height']
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
        self.avgpool3d = nn.AvgPool3d((kernel, width, height))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), #0.2 originally
            #nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        self.softmax = nn.Softmax(dim=1)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        #print(x.data.size()[-3:])
        x = self.avgpool3d(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        x = self.softmax(x)
        return x

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
    with open("/home/ctanama/framework/config2/mobilenetdriveprequant.yaml", "r") as ymlfile:
        #config = yaml.safe_load(ymlfile)
        config = yaml.load(ymlfile, Loader=yaml.Loader)
    #model = get_model(num_classes=600, sample_size = 112, width_mult=1.)
    #model = get_model(config, config['architecture'])
    config['train']["num_classes"] = config['pretraining']["model_num_classes"]
    model = MobileNet(config, config['architecture'])
    model = model.cuda()
    #model = nn.DataParallel(model, device_ids=None)
    #print(model)
    checkpoint = torch.load("/home/ctanama/pretrained/Pretrained-Models/kinetics_mobilenet_0.5x_RGB_16_best.pth")
    print(len(checkpoint['state_dict']))
    #model.load_state_dict(checkpoint['state_dict'])
    #print(model)

    print(sum(p.numel() for p in model.parameters()))
    new_dict = OrderedDict()

    for k, v in checkpoint['state_dict'].items():
        new_dict[k[7:]] = v
    #print([k for k,v in new_dict.items()])
    
    same = True

    #for k,v in checkpoint['state_dict'].items():
    #    if not torch.equal(checkpoint['state_dict'][k], new_dict[k[7:]]):
    #        same = False

    model.load_state_dict(new_dict)
    #print(same)

    #modules_to_fuse = [['features.0.0', 'features.0.1', 'features.0.2'], 
    #['features.1.conv1', 'features.1.bn1'], ['features.1.conv2', 'features.1.bn2', 'features.1.relu'],
    #['features.2.conv1', 'features.2.bn1'], ['features.2.conv2', 'features.2.bn2', 'features.2.relu'],
    #['features.3.conv1', 'features.3.bn1'], ['features.3.conv2', 'features.3.bn2', 'features.3.relu'],
    #['features.4.conv1', 'features.4.bn1'], ['features.4.conv2', 'features.4.bn2', 'features.4.relu'],
    #['features.5.conv1', 'features.5.bn1'], ['features.5.conv2', 'features.5.bn2', 'features.5.relu'],
    #['features.6.conv1', 'features.6.bn1'], ['features.6.conv2', 'features.6.bn2', 'features.6.relu'],
    #['features.7.conv1', 'features.7.bn1'], ['features.7.conv2', 'features.7.bn2', 'features.7.relu'],
    #['features.8.conv1', 'features.8.bn1'], ['features.8.conv2', 'features.8.bn2', 'features.8.relu'],
    #['features.9.conv1', 'features.9.bn1'], ['features.9.conv2', 'features.9.bn2', 'features.9.relu'],
    #['features.10.conv1', 'features.10.bn1'], ['features.10.conv2', 'features.10.bn2', 'features.10.relu'],
    #['features.11.conv1', 'features.11.bn1'], ['features.11.conv2', 'features.11.bn2', 'features.11.relu'],
    #['features.12.conv1', 'features.12.bn1'], ['features.12.conv2', 'features.12.bn2', 'features.12.relu'],
    #['features.13.conv1', 'features.13.bn1'], ['features.13.conv2', 'features.13.bn2', 'features.13.relu']]

    modules_to_fuse = config['train']['quantization']['fuse_module']
    print(model)

    fused_model = torch.quantization.fuse_modules(model, modules_to_fuse)

    print(fused_model)

    fused_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    torch.quantization.prepare_qat(fused_model, inplace=True)

    quantized_model = torch.quantization.convert(fused_model.to('cpu').eval(), inplace=False)

    checkpoint_best = torch.load("/home/ctanama/framework/model/StudentTeacherI3DMobileNetDrivePretrainingQuant/exp1/best_model.pth")

    quantized_model.load_state_dict(checkpoint_best['int_student_model_state_dict'])

    input_tensor = Variable(torch.randn(10, 3, 16, 224, 224))

    print(model(input_tensor.cuda()))

    print(quantized_model(input_tensor))

    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffers in model.buffers():
        buffer_size += buffers.nelement() * buffers.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.7f}MB'.format(size_all_mb))

    param_size_q = 0
    buffer_size_q = 0
    for param in quantized_model.parameters():
        param_size_q += param.nelement() * param.element_size()
    for buffers in quantized_model.buffers():
        buffer_size_q += buffers.nelement() * buffers.element_size()
    
    size_all_mb_q = (param_size_q + buffer_size_q) / 1024**2
    print('model size: {:.7f}MB'.format(size_all_mb_q))

    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

    torch.save(quantized_model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


    #input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    #output = model(input_var)
    #print(output.shape)
