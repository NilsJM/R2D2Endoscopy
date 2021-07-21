o# r2d2 model class
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


class r2d2(nn.Module):
    def __init__(self):
        super(r2d2, self).__init__()
        # input image to 32 channel
        # N: I think the input should be 3 not 1
        self.conv1 = conv_3x3_custom(3, 32) #torch.nn.Conv2d(1, 32, 3)
        
        self.conv2 = conv_3x3_custom(32, 32) # nn.Conv2d(32, 32, 3)
        self.conv3 = conv_3x3_custom(32, 32, dil=True) # nn.Conv2d(32, 64, 3, dilation=2)

        self.conv4 = conv_3x3_custom(32, 64) # nn.Conv2d(64, 64, 3)
        self.conv5 = conv_3x3_custom(64, 128, dil=True) # nn.Conv2d(64, 128, 3, dilation=2)

        self.conv6 = conv_3x3_custom(128, 128) # nn.Conv2d(128, 128, 3)
        self.conv7 = nn.Conv2d(128, 128, 2)
        self.conv8 = nn.Conv2d(128, 128, 2)
        self.conv9 = nn.Conv2d(128, 128, 2)
        
        self.conv10_1 = nn.Conv2d(128, 2, 1)
        self.conv11_1 = nn.Softmax(dim=1)  # I think it should be dim=2 or 3 depending on how it is counted (starting with 0 or 1 respectively)
        self.conv12_1 = nn.conv2d(2, 1, 1) # Not needed as the softmax is applied over that dimension
        self.conv10_2 = nn.Conv2d(128, 2, 1)
        self.conv11_2 = nn.Softmax(dim=1)
        self.conv12_2 = nn.conv2d(2, 1, 1) 
        
    # function returning a custum 3x3 conv block with option dilation
    def conv_3x3_custom(c_in, c_out, kernel, dil=False):
      if dil: 
        dilation = 2
      else: 
        dilation = 1
      
      return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, dilation=dilation),
        nn.BatchNorm2d(c_out),
        nn.ReLU()
      )

    def forward(self, inputs):
        # common layer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        # 3 seperate path
        descriptor = F.normalize(x, dim=0, p=2) # self.l2norm(x) also not sure which dimension
        x2 = x.square() #sqrt()
        x2_rep = self.conv10_1(x2)
        x2_rep = self.conv12_1(x2_rep)
        reliability = self.conv12_1(x2_rep) #= F.softmax(self.conv10_1(x2), dim=1) # who says the softmax is along dim=1?

        x2_rel = self.conv10_1(x2)
        x2_rel = self.conv12_1(x2_rel)
        reliability = self.conv12_1(x2_rel)

        return descriptor, reliability, repeatibility

    # function to get a patch p from 
    def get_patch(p, N=16): # DO WE HAVE A DEFAULT VALUE FOR N?

    # Cosine similarity cost function
    def cost_cosim(S, S_prime, U):
      P=
      S_primeU=
      1-(1/P)*np.sum(cosine_similarity(S,S_primeU))
      
    # Peakyines cost function
    def cost_peaky(I):

    # Repeatability cost function
    def cost_rep(I, I_primte, U, lambda=0.5):
      return cost_cosim(I, I_prime, U) + lambda * (cost_peaky(I) + cost_peaky(I_prime))

    def cost_reli(R, dims, kappa=0.5):
      L_AP = # Not sure how to initialize this variable in pytorch properly
      B = dims[0] * dims[1]

      for i in range(dims[0]):
        for j in range(dims[0]):
          L_AP += cost_reli_elmnt(R, i, j, kappa)
      
      return L_AP / B

    def cost_reli_elmnt(R, i, j, kappa=0.5):
      return 1 - (AvgPrecision(i, j) * R + kappa * (1 - R)) 
    
    # calculate the average prevision for current indices
    def AvgPrecision(i,j):
      return
