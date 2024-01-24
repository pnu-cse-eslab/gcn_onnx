from pyrsistent import v
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
import math



class alone(nn.Module):

    def __init__(self, base_model, in_channels, num_class, output_class, graph_args,
                 **kwargs):

        super().__init__()
        base_model = import_class(base_model)

        self.rad_stgcn = base_model(in_channels=in_channels, num_class=num_class,
                                        graph_args=graph_args,
                                        **kwargs)



        # self.fc_layer_1  = nn.Linear(num_class, output_class)
        self.fc_layer_1  = nn.Conv2d(num_class, output_class, 1)


        self.softmax = nn.Softmax(-1)



    def att(self, query, key, value):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        att_map = F.softmax(scores, dim=-1)
        return torch.matmul(att_map, value)                                


    def forward(self, first):

        rad_output = self.rad_stgcn(first) 
        #print("rad_output_____________out")
        #print(rad_output)
        #print(rad_output.shape)

        # att_output_rad = torch.sigmoid(rad_output)
        # #print("att_output_rad_____________out")
        # #print(att_output_rad)
        # #print(att_output_rad.shape)

        # fc_output = self.fc_layer_1(att_output_rad)

        
        # output = self.softmax(fc_output)
        
        # output = rad_output.squeeze(2)
        # # print(output.shape)
        
        # output = self.fc_layer_1(output)
        # # print(output.shape)
        # output = output.squeeze(2)

        # output = self.softmax(output)
        # # print(output.shape)
        # # print("_____________out")
        
        # ?????????????????
        # print(rad_output.shape)
        output = self.fc_layer_1(rad_output)
        # print(output.shape)
        output = output.squeeze(2).squeeze(2)
        # print(output.shape)
        output = self.softmax(output)
        # print(output.shape)
        # print("_____________out")

        return output, rad_output
    