from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy
import matplotlib.pyplot as plt
import pickle
from scipy import stats
import copy
from scipy.stats import entropy
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import mutual_info_score
import sys
from sklearn.cluster import KMeans

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
trainset = datasets.MNIST('data', download=True, train=True, transform=transform)
train_single =torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)
train_batch =torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1000, shuffle=True)
train_full = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=60000, shuffle=True)
test_full = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=10000, shuffle=True)
test_single = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)

imageNumberDataSet=[]
for i in range(10):
    imageNumberDataSet.append([])
for images,targets in train_full:
    for i in range(60000):
        imageNumberDataSet[targets[i].item()].append(images[i].detach().numpy())
for i in range(10):
    imageNumberDataSet[i]=numpy.array(imageNumberDataSet[i])
    print(i,imageNumberDataSet[i].shape)

testImageNumberDataSet=[]
for i in range(10):
    testImageNumberDataSet.append([])
for images,targets in test_full:
    for i in range(10000):
        testImageNumberDataSet[targets[i].item()].append(images[i].detach().numpy())
for i in range(10):
    testImageNumberDataSet[i]=numpy.array(testImageNumberDataSet[i])
    print(i,testImageNumberDataSet[i].shape)
    
def makeSingleTrainingSet(train_single,forNumber,n=5000):
    pos=0
    neg=0
    images=[]
    targets=[]
    for image,target in train_single:
        if target.item()==forNumber and pos<n:
            images.append(image.detach().numpy().reshape((28,28)))
            targets.append([1.0])
            pos+=1
        if target.item()!=forNumber and neg<n:
            images.append(image.detach().numpy().reshape((28,28)))
            neg+=1
            targets.append([-1.0])
    return numpy.array(images),numpy.array(targets)
    
def testOnAllTen(model,imageNumberDataSet,howMany=1000):
    images,targets,ids=makeCFtrainSet(imageNumberDataSet,[0,1,2,3,4,5,6,7,8,9],[howMany]*10,randomizeOrder=False)
    numberIDsOther=[[],[],[],[],[],[],[],[],[],[]]
    numberIDsThis=[[],[],[],[],[],[],[],[],[],[]]
    for i in range(10):
        for j in range(10):
            I=list(range(j*howMany,(j+1)*howMany))
            if j!=i:
                numberIDsOther[i]+=I
            else:
                numberIDsThis[i]+=I
    numberIDsThis=numpy.array(numberIDsThis)
    numberIDsOther=numpy.array(numberIDsOther)
    output=2.0*(model(torch.tensor(images)).detach().numpy().transpose()>0.0)-1.0
    C=list()
    targets=targets.transpose()
    for n in range(10):
        otherID=numpy.random.choice(numberIDsOther[n],(howMany),replace=False)
        TP=(1.0*(targets[n][numberIDsThis[n]]==output[n][numberIDsThis[n]])).sum()
        TN=(1.0*(targets[n][otherID]==output[n][otherID])).sum()
        C.append((TP+TN)/(howMany+howMany))
        #print(ids.shape)
    return C

def makeCFtrainSet(imageNumber,numbers,n,randomizeOrder=True,scrambleNumbers=[]):
    for i,number in enumerate(numbers):
        T=numpy.zeros((10,n[i]))-1.0
        T[number]=numpy.ones((n[i]))
        T=T.transpose()
        I=numpy.array([number]*n[i])
        which=numpy.random.choice(range(len(imageNumber[number])),(n[i]),replace=False)
        if number==numbers[0]:
            images=numpy.array(imageNumber[number][which])
            if number in scrambleNumbers:
                originalShape=images.shape
                images=images.flatten()
                numpy.random.shuffle(images)
                images=images.reshape(originalShape)
            targets=T
            ids=I
        else:
            toAdd=imageNumber[number][which]
            if number in scrambleNumbers:
                originalShape=toAdd.shape
                toAdd=toAdd.flatten()
                numpy.random.shuffle(toAdd)
                toAdd=toAdd.reshape(originalShape)
            images=numpy.concatenate((images,toAdd))
            targets=numpy.concatenate((targets,T))
            ids=numpy.concatenate((ids,I))
    if randomizeOrder:
        order=numpy.random.choice(range(images.shape[0]),(images.shape[0]),replace=False)
        images=images[order]
        targets=targets[order]
        ids=ids[order]
    return images,targets,ids
    
class ComposedNet(nn.Module):
    def __init__(self, input_dim=28*28,hidden_dim=50,output_dim=10,dropout_p=0.0):
        super(ComposedNet, self).__init__()
        self.hiddenLayer=nn.Linear(input_dim,hidden_dim)
        self.outputLayer=nn.Linear(hidden_dim,output_dim)
        self.output_dim=output_dim
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.do=nn.Dropout(dropout_p)
        
    def forward(self, x):
        out=x.view(-1,28*28)
        self.hidden=[]
        out=self.do(out)
        out=torch.tanh(self.hiddenLayer(self.do(out)))
        self.hidden.append(copy.deepcopy(out.detach().numpy()))
        #out=torch.sigmoid(self.outputLayer(out))
        out=torch.tanh(self.outputLayer(out))
        return out#F.log_softmax(out, dim=1)#out
        
            
    def induceDropout(self,p):
        self.hiddenLayer.weight.grad*=torch.Tensor(1.0*(numpy.random.random((self.hiddenLayer.weight.grad.shape))>p))
        self.hiddenLayer.bias.grad  *=torch.Tensor(1.0*(numpy.random.random((self.hiddenLayer.bias.grad.shape))>p))
        self.outputLayer.weight.grad*=torch.Tensor(1.0*(numpy.random.random((self.outputLayer.weight.grad.shape))>p))
        self.outputLayer.bias.grad*=torch.Tensor(1.0*(numpy.random.random((self.outputLayer.bias.grad.shape))>p))


kind=int(sys.argv[1])
p = float(sys.argv[2])
rep=int(sys.argv[3])
if kind==0:
    model=ComposedNet(28*28,20,10,dropout_p=p)
else:
    model=ComposedNet(28*28,20,10,dropout_p=0.0)

optimizer = optim.Adam(model.parameters(),lr=0.001)
error = nn.MSELoss()
#error = nn.CrossEntropyLoss()
#error = nn.L1Loss()
acc=0.0
Wtrain=[]
Wtest=[]
count=0
while acc<0.96:
    for image,target in train_batch:
        optimizer.zero_grad()
        targets=numpy.zeros((1000,10))-1.0
        for i,t in enumerate(target):
            targets[i][t]=1.0
        output = model(torch.Tensor(image))
        loss = error(output, torch.Tensor(targets))
        loss.backward()
        if kind==1:
            model.induceDropout(p=p)
        optimizer.step()
    Wtrain.append(testOnAllTen(model,imageNumberDataSet))
    acc=numpy.mean(Wtrain[-1])
    count+=1
    print(count,acc)
    if len(Wtrain)>5000:
        print("abort, retrain from scratch")
        if kind==0:
            model=ComposedNet(28*28,20,10,dropout_p=p)
        else:
            model=ComposedNet(28*28,20,10,dropout_p=0.0)
        acc=0.0
        Wtrain=[]
        Wtest=[]
        count=0
        optimizer = optim.Adam(model.parameters(),lr=0.001)
print(kind,p,rep,count,acc)
torch.save(model.state_dict(), "fullModel_MSE_k{0}_do{1}_rep{2}.model".format(kind,p,rep))
pickle.dump([Wtrain],open("Ws_fullModel_MSE_k{0}_do{1}_rep{2}.p".format(kind,p,rep),"wb"))