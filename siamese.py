#%%
import tensorflow as tf 
from joblib import Parallel, delayed
import utils
import os
import numpy as np
import glob
import random
import json
import sys
import math

# physical_devices = tf.config.experimental.list_physical_devices('DML')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.enable_eager_execution() 

print(tf.add([1.0, 2.0], [3.0, 4.0])) 


## 2028 = 169*12

#%%

def getFileNamesDict():
    print('Get File Names')
    fileList = []
    labelList = []
    # A = Atrial Premature Beat
    # F = Fusion of Ventricular beat and Normal Beat
    # j = Nodal Junction Escape Bead
    # N = Normal
    # R = Right Bundle Branch Block
    # S = Supraventricular Premature  or ectopic beat  Beat
    # V = Premature Ventricular Contraction
    beatTypes = ['A','S','F','j','N','R','V']
    for beatType in beatTypes:
        files = sorted([os.path.normpath(i).replace(os.sep,'/') for i in glob.glob('Dataset/'+beatType+'/*_'+'*.txt')])
        fileList = fileList + files
        labelList = labelList + [beatType for i in range(len(files))]
    zipped = list(zip(fileList,labelList))
    random.shuffle(zipped)
    fileList, labelList = zip(*zipped)
    return list(fileList),list(labelList)



def loadFilesDictParallel():
    fileNames,labels = getFileNamesDict()
    data = []
    dictData = {}   

    print('Load Files')
    data = Parallel(n_jobs=8)(delayed(np.loadtxt)(txt) for txt in fileNames)
    data = Parallel(n_jobs=8)(delayed(np.reshape)(xarr,(2028,1)) for xarr in data)
    for i,dataI in enumerate(data):
        if not labels[i] in dictData:
            dictData[labels[i]]= []
        dictData[labels[i]].append(dataI)
    print('Files Loaded')
    return dictData

# TODO - REFACTOR
def makeDataSets():
    print('Creating Datasets')
    trainFiles = []
    testFiles = []
    validationFiles = []
    trainLabels = []
    testLabels = []
    validationLabels = []
    for key in dictData.keys():
        files = dictData[key]
        valSize = math.floor(len(files)*0.15)
        validationFiles = validationFiles + files[:valSize]
        validationLabels = validationLabels + [key for i in range(valSize)]
        testSize = math.floor(len(files)*0.15)
        testFiles = testFiles + files[valSize:valSize+testSize]
        testLabels = testLabels + [key for i in range(testSize)]
        trainSize = len(files) - valSize - testSize
        trainFiles = trainFiles + files[valSize+testSize:]
        trainLabels = trainLabels + [key for i in range(trainSize)]

    trainFiles,trainLabels = shuffleFiles(trainFiles,trainLabels)
    testFiles,testLabels = shuffleFiles(testFiles,testLabels)
    validationFiles,validationLabels = shuffleFiles(validationFiles,validationLabels)
    print('Train Dataset')     
    trainDataset = tf.data.Dataset.from_tensor_slices((trainFiles,trainLabels))
    print('Test Dataset')
    testDataset = tf.data.Dataset.from_tensor_slices((testFiles,testLabels))
    print('Validation Dataset')
    validationDataset = tf.data.Dataset.from_tensor_slices((validationFiles,validationLabels))
    return trainDataset, testDataset,validationDataset

#TODO - Make It Faster
def processBatchDict(batch):
    for el in batch:
        x1 = []
        x2 = []
        y = []
        xEl = el[0]
        yEl = el[1]
        for i in range(0,len(el[0]),1):
            yEL = yEl[i].numpy().decode("utf-8").strip().replace('\0','')
            #Positive pair - Same Class
            positivePair = tf.convert_to_tensor(random.sample(dictData[yEL],1)[0],tf.float32)
            y.append(1)
            x1.append(xEl[i])
            x2.append(positivePair)
            nKey = random.sample([key for key in dictData.keys() if key != yEL],1)
            #Negative pair - Different Classes
            negativePair = tf.convert_to_tensor(random.sample(dictData[nKey[0]],1)[0],tf.float32)
            y.append(0)
            x1.append(xEl[i])
            x2.append(negativePair)

        yield np.array(x1,dtype=float),np.array(x2,dtype=float),np.array(y,dtype=int)


def shuffleFiles(files,labels):
    zipped = list(zip(files,labels))
    random.shuffle(zipped)
    filesList, labelsList = zip(*zipped)
    return list(filesList),list(labelsList)

def contrastive_loss(y, preds, margin=1):
	y = tf.cast(y, preds.dtype)
	
	squaredPreds = tf.keras.backend.square(preds)
	squaredMargin = tf.keras.backend.square(tf.keras.backend.maximum(margin - preds, 0))
	loss = tf.keras.backend.mean((1 - y) * squaredPreds + (y) * squaredMargin)

	return loss


def generateModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(16,7,activation=tf.keras.layers.LeakyReLU(),input_shape=(2028,1)))
    model.add(tf.keras.layers.Conv1D(32,5,activation=tf.keras.layers.LeakyReLU()))
    
    model.add(tf.keras.layers.MaxPool1D(pool_size=2,strides=2))
    
    model.add(tf.keras.layers.Conv1D(32,13,activation=tf.keras.layers.LeakyReLU()))
    model.add(tf.keras.layers.Conv1D(16,9,activation=tf.keras.layers.LeakyReLU()))
    
    model.add(tf.keras.layers.MaxPool1D(pool_size=2,strides=2))
    
    model.add(tf.keras.layers.Flatten())
   

    return model



model = generateModel()
model.summary()

print('Getting Files')
dictData = loadFilesDictParallel()
#%%
print('Generating Datasets')
trainDataset,testDataset,validationDataset = makeDataSets()
#%%
distFunction = utils.RMSE
distFunctionName = 'RMSE'


# Executes 10 Models (8h processing time ?)
k = 0
while k < 11:
    print(f'Model {k}')

    sig = tf.keras.layers.Input(shape=(2028,1))
    ref = tf.keras.layers.Input(shape=(2028,1))

    print('Generating Model')
    model = generateModel()
    model.summary()
    sigModel = model(sig)
    refModel = model(ref)
    
    # Define distance layer
    distance = tf.keras.layers.Lambda(distFunction)([sigModel, refModel])
    
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
    siameseModel = tf.keras.models.Model(inputs=[sig, ref], outputs=outputs)

    optmizer = tf.keras.optimizers.Adam()

    siameseModel.summary()
    siameseModel.compile(loss=contrastive_loss, optimizer=optmizer,
        metrics=["accuracy",tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TruePositives(),tf.keras.metrics.FalsePositives()])
    print('Initiating Training')
    epoch = 0
    loss = 0.0
    losses = []
    accs = []
    while epoch < 50: 
        print('Epoch ' +str(epoch+1))
        print('Shuffling')
        
        trainBatch = trainDataset.batch(64).prefetch(128).shuffle(10)
                    
        i = 1
        for x1,x2,y in processBatchDict(trainBatch):
            metric = siameseModel.train_on_batch([x1,x2],y,reset_metrics=True)
            if(i%10==0):
                i = 1
                print(f'LOSS:{metric[0]} ACC:{metric[1]} PRC:{metric[2]} RCL:{metric[3]} TN:{metric[4]} FN:{metric[5]}')
            i+=1

        # testFiles,testLabels = shuffleFiles(testFiles,testLabels)
        
        validationBatch = validationDataset.batch(64).prefetch(128)

        print('Validating Data')
        siameseModel.reset_metrics()
        for x1,x2,y in processBatchDict(validationBatch):
            metric = siameseModel.test_on_batch([x1,x2],y,reset_metrics=False)

        loss = metric[0]
        print('Validation:')
        print(f'LOSS:{metric[0]} ACC:{metric[1]} PRC:{metric[2]} RCL:{metric[3]} TN:{metric[4]} FN:{metric[5]}')
        siameseModel.save(f'Models/modelContrastive{distFunctionName}_{k}.h5')
        siameseModel.save_weights(f'Models/model_weightsContrastive{distFunctionName}_{k}.h5')
        
        losses.append(loss)
        accs.append(metric[1])
        
        
        with open(f'lossesContrastive{distFunctionName}_{k}.txt','w') as outfile:
            np.savetxt(outfile,losses)
        with open(f'accContrastive{distFunctionName}_{k}.txt','w') as outfile:
            np.savetxt(outfile,accs)

        epoch = epoch+1

    ##Testing
    print('Testing')
    testBatch = testDataset.batch(64).prefetch(256)
    siameseModel.reset_metrics()
    for x1,x2,y in processBatchDict(testBatch):
        metric = siameseModel.test_on_batch([x1,x2],y,reset_metrics=False)

    testResults = {'FP':metric[7],'FN':metric[5],'TP':metric[6],'TN':metric[4]}
    with open(f'TestResultsContrastiveRMSE_{k}.json','w') as outfile:
        json.dump(testResults,outfile)
    k+=1
# %%
