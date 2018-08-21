'''
quantized resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BinActive(torch.autograd.Function):
    """ Quantized activation operation

        Arguments:
            significant_bit: bit of quantization, default: 2-bit
            loss_regu: loss regularization term coefficient, currently not applied
    """    

    def __init__(self, significant_bit=2,loss_regu=0,alpha=1,pact_alpha=None):
        self.significant_bit = significant_bit
        self.threshold = pow(2, self.significant_bit)-1
        self.lb = 0
        self.rb = pow(2, self.significant_bit)-1
        

    def forward(self, input):
        self.save_for_backward(input)
        input_mod = torch.round(input.data)
        input_mod.data[input_mod.data.ge(self.rb)] = self.rb
        input_mod.data[input_mod.data.le(self.lb)] = self.lb
        return input_mod

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(self.rb)] = 0
        grad_input[input.le(self.lb)] = 0
        return grad_input

class BinActive_pact_sigmoid(torch.autograd.Function):
    """ Quantized activation operation using sigmoid esitmator to calculate gradient of step function

        Arguments:
            significant_bit: bit of quantization, default: 2-bit
            loss_regu: loss regularization term coefficient, currently not applied
            alpha: argument of scaled sigmoid function
            pact_alpha: a trainable clipping parameter in PACT
    """ 

    def __init__(self, significant_bit=2,loss_regu=0.01,alpha=1,pact_alpha=None):
        self.significant_bit = significant_bit
        self.threshold = pow(2, self.significant_bit)-1
        self.lb = 0.0
        self.rb = pow(2, self.significant_bit)-1.0
        self.loss_regu=loss_regu
        if pact_alpha is None:
            self.activation_threshold = torch.cuda.FloatTensor(1)
            self.activation_threshold[0] = self.rb
        else:
            self.activation_threshold = pact_alpha 
        # TODO: add function interface
        self.activation_lr = 1e-2
        self.activation_regu = 1e-4
        self.alpha = alpha

    def forward(self, input):
        input_mod = input.clone()
        self.save_for_backward(input)
        input_mod.data.mul_(self.rb/(1e-6+self.activation_threshold))
        input_mod.data = torch.round(input_mod.data)
        input_mod.data[input_mod.ge(self.rb)] = self.rb
        input_mod.data[input_mod.le(self.lb)] = self.lb
        input_mod.data.mul_((self.activation_threshold)/self.rb)
        return input_mod

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(self.activation_threshold[0])] = 0
        grad_input[input.le(0)] = 0

        alpha = self.alpha * (self.rb / self.activation_threshold[0])
        buf = torch.add(input*self.rb/self.activation_threshold[0],-0.5)-torch.round(torch.add(input*self.rb/self.activation_threshold[0],-0.5))
        buf.mul_(self.activation_threshold[0]/self.rb)
        temp_constant = 1/(2*(1/(1+math.exp(-1*alpha/2)))-1)
        grad = temp_constant*alpha*torch.exp(-1*alpha*buf)/torch.pow(torch.add(torch.exp(-1*alpha*buf),1),2)
        grad_input.data.mul_(grad/alpha) 
        """fix bug by dividing alpha on 08/21/2018, by WW """

        grad_alpha = grad_output.clone()
        grad_alpha[input.le(self.activation_threshold[0])] = 0
        grad_pact = torch.sum(grad_alpha)+2*self.activation_regu*self.activation_threshold[0]
        self.activation_threshold -= self.activation_lr * grad_pact

        return grad_input



class BinActive_pact(torch.autograd.Function):
    """ Implementation of quantized activation in PACT

        Arguments:
            significant_bit: bit of quantization, default: 2-bit
            loss_regu: loss regularization term coefficient, currently not applied
            alpha: argument of scaled sigmoid function
            pact_alpha: a trainable clipping parameter in PACT
    """ 

    def __init__(self, significant_bit=2,loss_regu=0.01,alpha=1,pact_alpha=None):
        super(BinActive_pact, self).__init__()
        self.significant_bit = significant_bit
        self.threshold = pow(2, self.significant_bit)-1
        self.lb = 0.0
        self.rb = pow(2, self.significant_bit)-1.0
        self.loss_regu=loss_regu
        if pact_alpha is None:
            self.activation_threshold = torch.cuda.FloatTensor(1)
            self.activation_threshold[0] = self.rb
        else:
            self.activation_threshold = pact_alpha
        self.activation_lr = 1e-2
        self.activation_regu = 1e-4

    def forward(self, input):
        # print(self.activation_threshold)        
        input_mod = input.clone()
        self.save_for_backward(input)
        input_mod.data.mul_(self.rb/(1e-6+self.activation_threshold))
        input_mod.data = torch.round(input_mod.data)
        input_mod.data[input_mod.ge(self.rb)] = self.rb
        input_mod.data[input_mod.le(self.lb)] = self.lb
        input_mod.data.mul_((self.activation_threshold)/self.rb)
        return input_mod
        
    def backward(self, grad_output):
        
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(self.activation_threshold[0])] = 0
        grad_input[input.le(0)] = 0
         
        grad_alpha = grad_output.clone()
        grad_alpha[input.le(self.activation_threshold[0])] = 0
        grad_pact = torch.sum(grad_alpha)+2*self.activation_regu*self.activation_threshold[0]
        self.activation_threshold -= self.activation_lr * grad_pact

        return grad_input
    

class BinActive_pact_exponential(torch.autograd.Function):
    """ Quantized activation, using exponent quantization

        Arguments:
            significant_bit: bit of quantization, default: 2-bit
            loss_regu: loss regularization term coefficient, currently not applied
            pact_alpha: a trainable clipping parameter in PACT
    """ 

    def __init__(self, significant_bit=2,loss_regu=0.01,alpha=1,pact_alpha=None):
        self.significant_bit = significant_bit
        self.threshold = pow(2, self.significant_bit)-1
        self.lb = 0.0
        self.rb = pow(2, self.significant_bit)-1.0
        self.loss_regu=loss_regu
        if pact_alpha is None:
            self.activation_threshold = torch.cuda.FloatTensor(1)
            self.activation_threshold[0] = self.rb
        else:
            self.activation_threshold = pact_alpha
        self.activation_lr = 1e-2
        self.activation_regu = 1e-4

    def forward(self, input):
        self.save_for_backward(input)
        input_mod = input.clone()
        activation_alpha = (1e-6+abs(self.activation_threshold[0]))/2**(self.rb)
        quantized_exponent = torch.round(torch.log(torch.abs(input_mod.data)/activation_alpha)/math.log(2.0))
        quantized_exponent[quantized_exponent>self.rb] = self.rb
        # quantized_exponent[quantized_exponent<self.lb] = self.lb
        input_mod = activation_alpha * 2**quantized_exponent

        return input_mod

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(self.activation_threshold[0])] = 0
        grad_input[input.le(0)] = 0
        grad_alpha = grad_output.clone()
        grad_alpha[input.le(self.activation_threshold[0])] = 0
        grad_pact = torch.sum(grad_alpha)+2*self.activation_regu*self.activation_threshold[0]
        self.activation_threshold -= self.activation_lr * grad_pact

        return grad_input


class BinActLayer(nn.Module):
    """ Customized activation layer quantization

        Arguments: 
            significant_bit: bit of activation layer quantization, default is 2
            loss_regu: additional regularization term coefficient
            alpha: argument of scaled sigmoid function, if applied
    """
    def __init__(self,significant_bit=2, loss_regu=0, alpha=1):
        super(BinActLayer, self).__init__()
        self.significant_bit = significant_bit
        self.loss_regu = loss_regu
        self.alpha = alpha
        self.pact_alpha = torch.cuda.FloatTensor(1)
        self.pact_alpha[0] = pow(2,significant_bit)-1

    def forward(self,input):
        return BinActive(significant_bit=self.significant_bit, loss_regu=self.loss_regu, alpha=self.alpha, pact_alpha=self.pact_alpha)(input)

class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, loss_regu=0.01, alpha=100,significant_bit=32, bit_threshold=10,  stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.binact1 = BinActLayer(significant_bit, loss_regu)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.binact2 = BinActLayer(significant_bit, loss_regu)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.significant_bit = significant_bit
        self.bit_threshold = bit_threshold
        self.use_sigmoid = True
        self.loss_regu = loss_regu

    def forward(self, x):
        residual = x
        x = self.binact1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.binact2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, quantize_bit, num_classes=10, alpha=100,loss_regu=0.01):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.loss_regu = loss_regu
        self.layer1 = self._make_layer(block, 16, layers[0], alpha=alpha, significant_bit=quantize_bit[0])
        self.layer2 = self._make_layer(block, 32, layers[1], alpha=alpha, stride=2, significant_bit=quantize_bit[1])
        self.layer3 = self._make_layer(block, 64, layers[2], alpha=alpha, stride=2, significant_bit=quantize_bit[2])
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.quantize_bit = quantize_bit

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, significant_bit=32,alpha=100):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
            #print(planes)

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample,alpha=alpha, significant_bit=significant_bit,loss_regu=self.loss_regu))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, significant_bit=significant_bit,alpha=alpha,loss_regu=self.loss_regu))

        return nn.Sequential(*layers)

    def forward(self, x):

        # Save full precision of x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Need quantization here, implemented in self.layer1-3
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Full precision for linear regression
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet_18(nn.Module):

    def __init__(self, block, layers, quantize_bit, num_classes=1000, alpha=100,loss_regu=0.01):
        super(ResNet_18, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.loss_regu = loss_regu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], alpha=alpha, significant_bit=quantize_bit[0])
        self.layer2 = self._make_layer(block, 128, layers[1], alpha=alpha, stride=2, significant_bit=quantize_bit[1])
        self.layer3 = self._make_layer(block, 256, layers[2], alpha=alpha, stride=2, significant_bit=quantize_bit[2])
        self.layer4 = self._make_layer(block, 512, layers[3], alpha=alpha, stride=2, significant_bit=quantize_bit[2])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.quantize_bit = quantize_bit

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, significant_bit=32,alpha=100):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
            #print(planes)

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample,alpha=alpha, significant_bit=significant_bit,loss_regu=self.loss_regu))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, significant_bit=significant_bit,alpha=alpha,loss_regu=self.loss_regu))

        return nn.Sequential(*layers)

    def forward(self, x):

        # Save full precision of x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Need quantization here, implemented in self.layer1-3
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Full precision for linear regression
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet20_cifar_quantized(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == '__main__':
    net = preact_resnet110_cifar()
    y = net(torch.autograd.Variable(torch.randn(1, 3, 32, 32)))
    print(net)
    print(y.size())

