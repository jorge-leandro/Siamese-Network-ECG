#%%

 
import os
import utils
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from joblib import Parallel, delayed
import glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
print(tf.version.VERSION)
def contrastive_loss(y, preds, margin=1):
	y = tf.cast(y, preds.dtype)
	
	squaredPreds = tf.keras.backend.square(preds)
	squaredMargin = tf.keras.backend.square(tf.keras.backend.maximum(margin - preds, 0))
	loss = tf.keras.backend.mean((1 - y) * squaredPreds + (y) * squaredMargin)

	return loss

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

# %%
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(16,7,activation=tf.keras.layers.LeakyReLU(),input_shape=(2028,1)))
model.add(tf.keras.layers.Conv1D(32,5,activation=tf.keras.layers.LeakyReLU()))

model.add(tf.keras.layers.MaxPool1D(pool_size=2,strides=2))

model.add(tf.keras.layers.Conv1D(32,13,activation=tf.keras.layers.LeakyReLU()))
model.add(tf.keras.layers.Conv1D(16,9,activation=tf.keras.layers.LeakyReLU()))

model.add(tf.keras.layers.MaxPool1D(pool_size=2,strides=2))

model.add(tf.keras.layers.Flatten())
#%%
sig = tf.keras.layers.Input(shape=(2028,1))
ref = tf.keras.layers.Input(shape=(2028,1))
sigModel = model(sig)
refModel = model(ref)
distFunction = utils.RMSE
distFunctionName = 'RMSE'
# Define distance layer
distance = tf.keras.layers.Lambda(distFunction)([sigModel, refModel])

outputs = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
siameseModel = tf.keras.models.Model(inputs=[sig, ref], outputs=outputs)

optmizer = tf.keras.optimizers.Adam()

siameseModel.summary()
siameseModel.compile(loss=contrastive_loss, optimizer=optmizer,
    metrics=["accuracy",tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TruePositives(),tf.keras.metrics.FalsePositives()])
# %%
siameseModel.load_weights('Models/model_weightsContrastiveRMSE_1.h5')

# %%
# dataFiles = np.load('dataFiles.npy')
# dataLabels = np.load('dataLabels.npy')
dictData = loadFilesDictParallel()

dataFiles = []
dataLabels = []
for key in dictData.keys():
    dataFiles = dataFiles + dictData[key]
    dataLabels = dataLabels + [key for i in range(len(dictData[key]))]

validationSize = .2
trainValidationFiles, testFiles, trainValidationLabels, testLabels = train_test_split(dataFiles,dataLabels,test_size=validationSize,random_state=42)

testFiles,testLabels = shuffleFiles(testFiles,testLabels)
testDataset = tf.data.Dataset.from_tensor_slices((testFiles,testLabels))

#%%
testBatch = testDataset.batch(64).prefetch(256)

#%% 
onlyfiles = [f for f in os.listdir('Models') if os.path.isfile(os.path.join('Models', f))]
modelsWeights = [f for f in onlyfiles if 'model_weightsContrastive' in f]
modelsWeights.sort()

# %%
index = 0
results = {}
for weights in modelsWeights:
    siameseModel.load_weights('Models/'+weights)
    siameseModel.reset_metrics()

    y_test = []
    y_proba = []

    for x1,x2,y in processBatchDict(testBatch):
        predicted = siameseModel.predict([x1,x2], verbose=0)
        y_test = y_test + y.tolist()
        y_proba = y_proba + predicted.tolist()
        metric = siameseModel.test_on_batch([x1,x2],y,reset_metrics=False)

    testResults = {'FP':metric[7],'FN':metric[5],'TP':metric[6],'TN':metric[4]}
    n = len(y_test)
    ratio = .95
    n_0 = int((1-ratio) * n)
    n_1 = int(ratio * n)
    y_pred = np.array(y_proba) > .5
    results[index] = { 'y_test':y_test,'y_proba':y_proba,'y_pred':y_pred,'testResults':testResults }
    print(f'accuracy score: {accuracy_score(y_test, y_pred)}')
    cf_mat = confusion_matrix(y_test, y_pred)
    print(cf_mat)
    plot_roc_curve(y_test, y_proba)
    print(f'model {index} AUC score: {roc_auc_score(y_test, y_proba)}')
    index = index + 1

 # %%

AUCs = []

for key in results.keys():
    AUCs.append(roc_auc_score(results[key]['y_test'], results[key]['y_proba']))

# %%
