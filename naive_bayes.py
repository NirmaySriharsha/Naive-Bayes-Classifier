import numpy as np
import math
import matplotlib.pyplot as plt
def NB_GenerateData(D, p, count):
    
    '''
    This function generates data to follow the probabilities that the Naive Bayes model learned
    D is the matrix of P(X|Y) probabilities
    p is the P(Y) prior probabilities
    count is the number of data points we will generate
    '''
    xData = np.zeros((count, D.shape[1]))
    countTrue = np.sum(np.random.binomial(1, p, count))
    countFalse = count-countTrue
    trueY = np.ones(countTrue)+1
    falseY = np.ones(countFalse)
    yData = np.concatenate((trueY, falseY))
    for f in range(D.shape[1]):
        trueX = np.random.binomial(1,D[1][f],countTrue)
        falseX = np.random.binomial(1,D[0][f],countFalse)
        tempX = np.concatenate((trueX, falseX))
        xData[:,f] = tempX
    return xData, yData

def augmentFeatures(XTrain, yTrain, XTest, yTest):
    te_vals = {}
    teaug_vals = {}

    np.random.seed(0)
    for i in range(150,451,30):
        te_vals[i] = 0
        teaug_vals[i] = 0

    for i in range(10):
        fset = np.random.choice(XTrain.shape[1],50,replace=False)

        for m in range(150,451,30):
            D = NB_XGivenY(XTrain[0:m], yTrain[0:m])
            p = NB_YPrior(yTrain[0:m])

            yHatTest = NB_Classify(D, p, XTest)

            testAcc = NB_ClassificationAccuracy(yHatTest, yTest)

            te_vals[m] += testAcc
            
            #augmebting features
            xtrain_aug = np.repeat(XTrain[0:m,fset],100,axis=1)
            xtest_aug = np.repeat(XTest[:,fset],100,axis=1)
            xTrain = np.hstack([XTrain[0:m],xtrain_aug])
            xTest = np.hstack([XTest,xtest_aug])
            D = NB_XGivenY(xTrain, yTrain[0:m])
            p = NB_YPrior(yTrain[0:m])

            yHatTest = NB_Classify(D, p, xTest)

            testAcc = NB_ClassificationAccuracy(yHatTest, yTest)
            teaug_vals[m] += testAcc

            

    for i in range(150,451,30):
        te_vals[i] /= 10
        teaug_vals[i] /= 10

    plt.plot(list(te_vals.keys()),list(te_vals.values()),label='Original dataset')
    plt.plot(list(teaug_vals.keys()),list(teaug_vals.values()),label='With augmented features')
    plt.legend()
    plt.xlabel('m')
    plt.ylabel('Test Accuracy')
    plt.savefig('./q9.pdf')


def NB_XGivenY(XTrain, yTrain, a=0.001, b=0.9):

    THRESHOLD = 1e-5
    yTrain_better = yTrain - 1 #Dealing with the whole 1 and 2 thing is annoying; I'm making it 0 and 1 instead. 
    Econ_Indexes =  np.squeeze(1 - yTrain_better)
    Onion_Indexes = np.squeeze(yTrain_better)
    Indexes = np.squeeze([Econ_Indexes, Onion_Indexes])
    Values = Indexes@XTrain
    Thetas_Intermediate = Values + a
    Thetas_Intermediate[0,:] = Thetas_Intermediate[0,:]/(a+b+np.sum(Econ_Indexes))
    Thetas_Intermediate[1,:] = Thetas_Intermediate[1,:]/(a+b+np.sum(Onion_Indexes))
    Thetas = np.clip(Thetas_Intermediate, THRESHOLD, 1 - THRESHOLD)
    return Thetas 

     


    # TODO: Implement P(X|Y) with Beta(a,1+b) Prior


def NB_YPrior(yTrain):
    p = np.sum(2-yTrain)/len(yTrain)
    return p

    # TODO: Implement P(Y)


def NB_Classify(D, p, X):
    num_rows, num_cols = X.shape
    D_1 = np.squeeze([D[0,:],]*num_rows)
    D_2 = np.squeeze([D[1,:],]*num_rows)
    Intermediate_1 = np.abs(D_1 + (X-1))
    Intermediate_2 = np.abs(D_2 + (X-1))
    Intermediate_1 = np.log(Intermediate_1)
    Intermediate_2 = np.log(Intermediate_2)
    K = np.full((num_cols, 1),1)
    Y_1 = Intermediate_1@K + np.log(p)
    Y_2 = Intermediate_2@K + np.log(1-p)
    YHat = (Y_1 <= Y_2).astype(int) + 1
    return YHat
    
    # TODO: Implement Label Predition Vector of X


def NB_ClassificationAccuracy(yHat, yTruth):
    count = 0
    #I got lazy at this point so brute forced it. 
    for i in range(0, len(yTruth)):
        if yHat[i] == yTruth[i]:
            count = count + 1
    return count/len(yTruth)
    # TODO: Compute the Classificaion Accuracy of yHat Against yTruth

if __name__ == "__main__":
    import pickle
    with open("hw1data.pkl", "rb") as f:
        data = pickle.load(f)
    
    XTrain = data["XTrain"]
    YTrain = data["yTrain"]
    XTest = data["XTest"]
    yTest = data["yTest"]
    Vocab = data["Vocabulary"] 




#5.6 Code
#m = [i for i in range(100,450,30)]
#m.append(450)
#Training_Acc = []
#Test_Acc = []
#for i in m:
#    D = NB_XGivenY(XTrain[0:i], YTrain[0:i])
#    p = NB_YPrior(YTrain[0:i])
#    Training_Acc.append(NB_ClassificationAccuracy(NB_Classify(D,p,XTrain),YTrain))
#    Test_Acc.append(NB_ClassificationAccuracy(NB_Classify(D,p,XTest),yTest))
#
#plt.plot(m, Training_Acc, label = "Training Accuracy")
#plt.plot(m,Test_Acc, label = "Test Accuracy")
#plt.legend()
#plt.xlabel("Size of Initial Slice")
#plt.ylabel("Accuracy")
#plt.show()

#5.7 Code
#D = NB_XGivenY(XTrain, YTrain, a = 0.001, b =0.9)
#p = NB_YPrior(YTrain)
#D_0 = D[0,:]
#D_1 = D[1,:]
#print(Vocab[D_0.argsort()[::-1][:5]])
#print(Vocab[D_1.argsort()[::-1][:5]])
#E_0 = D_0/D_1
#E_1 = D_1/D_0
#print(Vocab[E_0.argsort()[::-1][:5]])
#print(Vocab[E_1.argsort()[::-1][:5]])

#5.8 Code
#D = NB_XGivenY(XTrain, YTrain, a = 0.001, b = 0.9)
#p = NB_YPrior(YTrain)
#M_1 = np.array([-1,1])
#M_2 = np.array([1,-1])
#Threshholds = [i/10 for i in range(0,11)]
#Num_Words_Deleted = [0]
#Training_Acc = []
#Test_Acc = []
#for i in Threshholds:
#    #I_01 = np.abs(M_1@D/D[1,:])
#    #Indices = I_01 < i
#    I_10 = np.abs(M_2@D/D[0,:])
#    Indices = I_10 < i
#    S_1 = np.sum(Indices.astype(int))
#    if i >0: 
#        Num_Words_Deleted.append(S_1+Num_Words_Deleted[len(Num_Words_Deleted)-1])
#    #D = np.delete(D, Indices, 1)
#    XTrain = np.delete(XTrain, Indices, 1)
#    XTest = np.delete(XTest, Indices, 1)
#    D = NB_XGivenY(XTrain, YTrain, a = 0.001, b =0.9)
#    Training_Acc.append(NB_ClassificationAccuracy(NB_Classify(D,p,XTrain),YTrain))
#    Test_Acc.append(NB_ClassificationAccuracy(NB_Classify(D,p,XTest), yTest))
#plt.plot(Threshholds, Training_Acc, label = "Training Accuracy")
#plt.plot(Threshholds, Test_Acc, label = "Test Accuracy")
#plt.legend()
#plt.xlabel("Threshhold")
#plt.ylabel("Accuracy")
##plt.title("Accuracies under I_01 metric")
#plt.title("Accuracies under I_10 metric")
#plt.show()
#plt.plot(Num_Words_Deleted, Training_Acc)
#plt.plot(Num_Words_Deleted, Test_Acc)
#plt.show()
#plt.plot(Threshholds, Num_Words_Deleted)
#plt.show()

#5.9 Code
#augmentFeatures(XTrain, YTrain, XTest, yTest)

#5.10 Code
D = NB_XGivenY(XTrain, YTrain, a = 0.001, b = 0.9)
p = NB_YPrior(YTrain)
XData, YData = NB_GenerateData(D,p,1000)
print(NB_ClassificationAccuracy(NB_Classify(D,p,XData),YData))