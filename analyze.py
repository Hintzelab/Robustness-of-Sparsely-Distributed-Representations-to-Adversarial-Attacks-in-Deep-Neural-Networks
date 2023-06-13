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


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def testFSGM( model, test_loader, epsilon ,verbose=False,device="cpu"):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    if verbose:
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
    
def symbolEntropy(D,base=2):
    value,counts = numpy.unique(D, return_counts=True)
    return entropy(counts,base=base)

def RwithoutS(I,H,Hmask,iMult=2):
    A=I
    B=numpy.bitwise_and(H,Hmask)
    AB=B*iMult+I
    hA=symbolEntropy(A,base=2)
    hB=symbolEntropy(B,base=2)
    hAB=symbolEntropy(AB,base=2)
    return hA+hB-hAB
    
def getOutTaH(model,dataSet):
    for batch_idx, (d, t) in enumerate(dataSet):
        o = model(d)#+torch.Tensor(numpy.random.uniform(0.0,0.1,d.shape)))
        h=model.hidden
        data=d
        target=t.detach().numpy()
        output=o.detach().numpy()
        hidden=numpy.array(model.hidden[0])

        A=hidden.transpose()
        
    B=numpy.zeros(A.shape)
    clusterNr=2
    for i in range(B.shape[0]):
        a=A[i].reshape(-1,1)
        if len(numpy.unique(a))==1:
            who=numpy.random.randint(len(a))
            a[who]=1-a[who]
        kmeans = KMeans(n_clusters=clusterNr).fit(a)
        B[i]=kmeans.labels_
        #B[i]=1.0*(A[i]>numpy.median(A[i]))


    H=numpy.zeros((target.shape))
    for i in range(20):
        H+=B[i]*(2**i)
    H=H.astype((int))
    return output,target,H
    
def singleShrinkingDecompositionInformation(I,H,width=20):
    nodes=list(range(width))
    cols=[]
    colh=[]
    while len(nodes)>0:
        infos=[]
        for node in nodes:
            subset=copy.deepcopy(nodes)
            subset.remove(node)
            mask=0
            for s in subset:
                mask+=1*(2**s)
            mask=int(mask)
            h=RwithoutS(I,H,mask)
            infos.append(h)
        nodeToDrop=nodes[infos.index(max(infos))]
        nodes.remove(nodeToDrop)
        cols.append(copy.deepcopy(nodes))
        colh.append(max(infos))
    return cols,colh

def shrinkingDecompositionInformation(model,width,dataSet,numbers=[0,1,2,3,4,5,6,7,8,9]):
    output,target,H=getOutTaH(model,dataSet)
    collectorSet=dict()
    collectorH=dict()
    for number in numbers:
        I=(1.0*(target==number)).astype(int)
        s,h=singleShrinkingDecompositionInformation(I,H,width)
        collectorSet[number]=s
        collectorH[number]=h
    return collectorSet,collectorH

def removalIntoVec(res,width,H):
    V=numpy.zeros(width)
    for i,r in enumerate(res):
        for e in r:
            V[e]+=H[0]-H[i]

    #V=sqrt(V)
    if V.sum()==0:
        return V
    return V#/V.max()

def removalIntoMatrix(res,width,H):
    M=[]
    for i in range(10):
        M.append(removalIntoVec(res[i],width,H[i]))
    return numpy.array(M)

def smearedness(M):
    S=[]
    for i in range(20):
        S.append(sort(M.transpose()[i])[:-1].sum())
    return numpy.array(S)



kind=int(sys.argv[1])
p = float(sys.argv[2])
rep=int(sys.argv[3])
if kind==0:
    model=ComposedNet(28*28,20,10,dropout_p=p)
else:
    model=ComposedNet(28*28,20,10,dropout_p=0.0)

weights=torch.load( "fullModel_MSE_k{0}_do{1}_rep{2}.model".format(kind,p,rep))
model.load_state_dict(weights)
model.eval()

epsilons = [0, .05, .1, .15, .2, .25, .3]
accuracies=[]
for eps in epsilons:
    acc, ex = testFSGM(model, test_single, eps)
    accuracies.append(acc)
pickle.dump(accuracies,open("accFSGM_MSE_k{0}_do{1}_rep{2}.p".format(kind,p,rep),"wb"))

S,H=shrinkingDecompositionInformation(model,20,test_full)
M=removalIntoMatrix(S,20,H)
pickle.dump([S,H,M],open("SHM_MSE_k{0}_do{1}_rep{2}.p".format(kind,p,rep),"wb"))
