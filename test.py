
import numpy as np
import paddle
import torch
import os
from config_cvt.default import get_config
import torchvision.models as models
from cvt import generate_model
from cvt_torch import get_cls_model
model_name = 'CvT-13-224x224-IN-1k.pth'

config=get_config('./config_cvt/config.yaml')
paddle.set_device('cpu')
paddle_model = generate_model(config)
paddle_model.eval()


device = torch.device('cpu')
torch_model= get_cls_model(config)
torch_model.load_state_dict= torch.load(model_name,map_location=torch.device('cpu'))

torch_model = torch_model.to(device)
torch_model.eval()
file_debug=open('debug.txt','w')
def torch_to_paddle_mapping(torch_model,paddle_model):
    paddle_array=[]
    torch_array=[]
    for i,value in torch_model.state_dict().items():
        torch_array.append((i,value.shape))
    for i,value in paddle_model.state_dict().items():
        paddle_array.append((i,value.shape))

    j=0
    mapping=[]
    def equel(i,j):
        x:str=torch_array[i][0]
        y:str=paddle_array[j][0]
        if x==y:
            return True
        x=x.replace('.bn.','.1.')
        x=x.replace('.conv.','.0.')
        x=x.replace('.running_mean','._mean')
        x=x.replace('.running_var','._variance')
        if x==y:
            return True
        else:
            return False
    for i in range(len(torch_array)):
        if equel(i,j):
            mapping.append((torch_array[i][0],paddle_array[j][0]))
            j+=1
    if len(mapping)!=len(paddle_array):
        assert RuntimeError(f'mapping is not full,length of mapping is{len(mapping)},length of paddle_array is {len(paddle_array)}')
    return mapping
def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name, transpose=True):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) 

        file_debug.write(f'**SET** {th_name} {th_shape} **TO** {pd_name} {pd_shape}')
        file_debug.write('\n')
        if isinstance(th_params[th_name], torch.nn.parameter.Parameter):
            value = th_params[th_name].data.numpy()
        else:
            value = th_params[th_name].numpy()

        if len(value.shape) == 2 and transpose:
            value = value.transpose((1, 0))
        pd_params[pd_name].set_value(value)

    # 1. get paddle and torch model parameters
    pd_params = {}
    th_params = {}
    for name, param in paddle_model.named_parameters():
        pd_params[name] = param
    for name, param in torch_model.named_parameters():
        th_params[name] = param

    for name, param in paddle_model.named_buffers():
        pd_params[name] = param
    for name, param in torch_model.named_buffers():
        th_params[name] = param

    # 2. get name mapping pairs
    mapping = torch_to_paddle_mapping(torch_model,paddle_model)

    # 3. set torch param values to paddle params: may needs transpose on weights
    for th_name, pd_name in mapping:
        if th_name in th_params.keys(): # nn.Parameters
            _set_value(th_name, pd_name)
        else: # weight & bias
            th_name_w = f'{th_name}.weight'
            pd_name_w = f'{pd_name}.weight'
            _set_value(th_name_w, pd_name_w)

            if f'{th_name}.bias' in th_params.keys():
                th_name_b = f'{th_name}.bias'
                pd_name_b = f'{pd_name}.bias'
                _set_value(th_name_b, pd_name_b)

    return paddle_model
# convert weights
paddle_model = convert(torch_model, paddle_model)

# check correctness
x = np.random.randn(2, 3, 224, 224).astype('float32')
x_paddle = paddle.to_tensor(x)
x_torch = torch.Tensor(x)
paddle.set_printoptions(4)
torch.set_printoptions(4)
out_torch = torch_model(x_torch)
out_paddle = paddle_model(x_paddle)




out_paddle = paddle_model(x_paddle)

out_torch = out_torch.data.cpu().numpy()
out_paddle = out_paddle.cpu().numpy()

file_debug.write(str(out_torch.shape))
file_debug.write('\n')
file_debug.write( str(out_paddle.shape))
file_debug.write('\n')
file_debug.write(str(out_torch[0, 0:100]))
file_debug.write('\n')
file_debug.write('========================================================')
file_debug.write('\n')
file_debug.write(str(out_paddle[0, 0:100]))
file_debug.write('\n')

assert np.allclose(out_torch, out_paddle, atol = 1e-5)

# save weights for paddle model
model_path = os.path.join(f'./{model_name}.pdparams')
paddle.save(paddle_model.state_dict(), model_path)
file_debug.write('all done')
