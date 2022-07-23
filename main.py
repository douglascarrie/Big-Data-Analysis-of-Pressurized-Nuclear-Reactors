
# IMPORTS
import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
import findspark
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import DecisionTreeClassifier, MultilayerPerceptronClassifier, LinearSVC
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator




# finds spark module // so its ready to use.


# create spark session
spark = SparkSession.builder.getOrCreate()
# make own schema as provided header has some strange spaces.
schema = StructType([
    StructField("Status", StringType(), True),
    StructField("Power_range_sensor_1", DoubleType(), True),
    StructField("Power_range_sensor_2", DoubleType(), True),
    StructField("Power_range_sensor_3", DoubleType(), True),
    StructField("Power_range_sensor_4", DoubleType(), True),
    StructField("Pressure_sensor_1", DoubleType(), True),
    StructField("Pressure_sensor_2", DoubleType(), True),
    StructField("Pressure_sensor_3", DoubleType(), True),
    StructField("Pressure_sensor_4", DoubleType(), True),
    StructField("Vibration_sensor_1", DoubleType(), True),
    StructField("Vibration_sensor_2", DoubleType(), True),
    StructField("Vibration_sensor_3", DoubleType(), True),
    StructField("Vibration_sensor_4", DoubleType(), True),


])

# no header as using above structure
df = spark.read.csv("nuclear_plants_small_dataset.csv", header=False, schema=schema )

# show schema     
df.printSchema()





'''

Checking for NULL or NAN Values

'''
from pyspark.sql.functions import col,isnan, when, count, rand

#slect null values
df.na.drop("all")


'''

Seperating Normal and abnormal data into different df

'''

# Seperate all Normal status values into new dataframe
dfNormal = df.filter(df.Status == 'Normal' )


# Seperate all Abnormal status values into new dataframe
dfAbnormal = df.filter(df.Status == 'Abnormal' )


#dfNormal.show(vertical=True)
#dfAbnormal.show(vertical=True)



'''
Getting Maximum

'''

# sensors variable will be used for all methods (max,min,mode,etc)
sensors = ["Power_range_sensor_", "Pressure_sensor_", "Vibration_sensor_"]
for x in sensors:
    i = 1
    print(f'NORMAL --MAX - {x}1-4\n')
    while i < 5:
        print(dfNormal.agg({x+ str(i): 'max'}).show(vertical=True))
        i += 1
    i = 1
    print(f'ABNORMAL --MAX - {x}1-4\n')
    while i < 5:

      print(dfAbnormal.agg({x+ str(i): 'max'}).show(vertical=True),'\n')
      i += 1



'''
Getting Minimum

'''


for x in sensors:
    i = 1
    print(f'NORMAL --MIN - {x}1-4\n')
    while i < 5:

      print(dfNormal.agg({x+ str(i): 'min'}).show(vertical=True))
      i += 1
    i = 1
    print(f'ABNORMAL --MIN - {x}1-4\n')
    while i < 5:

      print(dfAbnormal.agg({x+ str(i): 'min'}).show(vertical=True))
      i += 1


'''

Getting Mean

'''

for x in sensors:
    i = 1
    print(f'NORMAL --MEAN - {x}1-4\n')
    while i < 5:

      print(dfNormal.agg({x+ str(i): 'mean'}).show(vertical=True))
      i += 1
    i = 1
    print(f'ABNORMAL --MEAN - {x}1-4\n')
    while i < 5:

      print(dfAbnormal.agg({x+ str(i): 'mean'}).show(vertical=True))
      i += 1



'''
Getting Variance

'''

for x in sensors:
    i = 1
    print(f'NORMAL --Variance - {x}1-4\n')
    while i < 5:

      print(dfNormal.agg({x+ str(i): 'variance'}).show(vertical=True))
      i += 1
    i = 1
    print(f'ABNORMAL --Variance - {x}1-4\n')
    while i < 5:

      print(dfAbnormal.agg({x+ str(i): 'variance'}).show(vertical=True))
      i += 1




'''

Converting Pyspark df to Pandas df

'''

PdfNormal = dfNormal.toPandas()
PdfAbnormal = dfAbnormal.toPandas()





'''

Getting Median

'''

for x in sensors:
    i = 1
    print(f'\nNORMAL --Median - {x}1-4\n')
    while i < 5:

      print(x+str(i)+': ',PdfNormal['Vibration_sensor_'+str(i)].median())
      i += 1
    i = 1
    print(f'\nABNORMAL --Median - {x}1-4\n')
    while i < 5:

      print(x+str(i)+': ',PdfAbnormal['Vibration_sensor_'+str(i)].median())
      i += 1




'''
Getting Mode

'''
for x in sensors:
    i = 1
    print(f'\nNORMAL --Mode - {x}1-4\n')
    while i < 5:

      print(x+str(i)+': ',PdfNormal['Power_range_sensor_'+str(i)].mode()[0])
      i += 1
    i = 1
    print(f'\nABNORMAL --Mode - {x}1-4\n')
    while i < 5:

      print(x+str(i)+': ',PdfAbnormal['Power_range_sensor_'+str(i)].mode()[0])
      i += 1



'''
Making boxplots of the data

using seaborn

'''
#set style for boxplot
sns.set_theme(style='whitegrid')

#Normal - POWER RANGE SENSORS
xN = PdfNormal[["Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3", "Power_range_sensor_4"]].rename(columns={"Power_range_sensor_1" :"Sensor 1", "Power_range_sensor_2" : "Sensor 2", "Power_range_sensor_3" : "Sensor 3","Power_range_sensor_4" : "Sensor 4"})
sns.boxplot(data=xN).set_title('Power Range Sensors -- Status: NORMAL')
sns.despine(left=True)
plt.savefig('Boxplots/Power_range_sensors_Normal.png',dpi=180)


#AbNormal - POWER RANGE SENSORS
xA = PdfAbnormal[["Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3", "Power_range_sensor_4"]].rename(columns={"Power_range_sensor_1" :"Sensor 1", "Power_range_sensor_2" : "Sensor 2", "Power_range_sensor_3" : "Sensor 3","Power_range_sensor_4" : "Sensor 4"})
sns.boxplot(data=xA).set_title('Power Range Sensors -- Status: ABNORMAL')
sns.despine(left=True)
plt.savefig('Boxplots/Power_range_sensors_Abnormal.png',dpi=180)

#Normal - PRESSURE SENSORS
xA = PdfNormal[["Pressure_sensor_1", "Pressure_sensor_2", "Pressure_sensor_3", "Pressure_sensor_4"]].rename(columns={"Pressure_sensor_1" :"Sensor 1", "Pressure_sensor_2" : "Sensor 2", "Pressure_sensor_3" : "Sensor 3", "Pressure_sensor_4" : "Sensor 4"})
sns.boxplot(data=xA).set_title('Pressure Sensors -- Status: NORMAL')
sns.despine(left=True)
plt.savefig('Boxplots/Pressure_sensors_Normal.png',dpi=180)

#Abnormal - PRESSURE SENSORS
xA = PdfAbnormal[["Pressure_sensor_1", "Pressure_sensor_2", "Pressure_sensor_3", "Pressure_sensor_4"]].rename(columns={"Pressure_sensor_1" :"Sensor 1", "Pressure_sensor_2" : "Sensor 2", "Pressure_sensor_3" : "Sensor 3", "Pressure_sensor_4" : "Sensor 4"})
sns.boxplot(data=xA).set_title('Pressure Sensors -- Status: ABNORMAL')
sns.despine(left=True)
plt.savefig('Boxplots/Pressure_sensors_Abnormal.png',dpi=180)

#Normal - VIBRATION SESNORS 
xA = PdfNormal[["Vibration_sensor_1", "Vibration_sensor_2", "Vibration_sensor_3", "Vibration_sensor_4"]].rename(columns={"Vibration_sensor_1" :"Sensor 1", "Vibration_sensor_2" : "Sensor 2", "Vibration_sensor_3" : "Sensor 3", "Vibration_sensor_4" : "Sensor 4"})
sns.boxplot(data=xA).set_title('Vibration Sensors -- Status: NORMAL')
sns.despine(left=True)
plt.savefig('Boxplots/Vibration_sensors_Normal.png',dpi=180)

#Abnormal - VIBRATION SESNORS 
xA = PdfAbnormal[["Vibration_sensor_1", "Vibration_sensor_2", "Vibration_sensor_3", "Vibration_sensor_4"]].rename(columns={"Vibration_sensor_1" :"Sensor 1", "Vibration_sensor_2" : "Sensor 2", "Vibration_sensor_3" : "Sensor 3", "Vibration_sensor_4" : "Sensor 4"})
sns.boxplot(data=xA).set_title('Vibration Sensors -- Status: ABNORMAL')
sns.despine(left=True)
plt.savefig('Boxplots/Vibration_sensors_Abnormal.png',dpi=180)




'''
Correlation matrix

using seaborn

'''
dfP = df.toPandas()
dfP = dfP.drop('Status', axis=1)
plt.figure(figsize=(16, 12))

ax = sns.heatmap(dfP.corr(), annot=True, vmin=-1, vmax=1)
ax.set_title("Correlation Matrix of the Features",fontsize = 23)
plt.savefig('CorrelationMatrix.png',dpi=120)





'''

SECTION 2

'''



#convert sensor type to Integers from Doubles
for x in range(1, 5):

    df = df.withColumn('Power_range_sensor_'+str(x) , df['Power_range_sensor_'+str(x)].cast(IntegerType()))
    df = df.withColumn('Pressure_sensor_'+str(x) , df['Pressure_sensor_'+str(x)].cast(IntegerType()))
    df = df.withColumn('Vibration_sensor_'+str(x) , df['Vibration_sensor_'+str(x)].cast(IntegerType()))
    


#check they have changed
df.printSchema()


'''
Build vector assembler for models

'''
assembler = VectorAssembler(
    inputCols=[
 'Power_range_sensor_1',
 'Power_range_sensor_2',
 'Power_range_sensor_3',
 'Power_range_sensor_4',
 'Pressure_sensor_1',
 'Pressure_sensor_2',
 'Pressure_sensor_3',
 'Pressure_sensor_4',
 'Vibration_sensor_1',
 'Vibration_sensor_2',
 'Vibration_sensor_3',
 'Vibration_sensor_4'

    ],
    outputCol='features',
    

)
# used to skip null entires // needed as when fixing header names, it creates some null collum apart from status which could not be removed
assembler.setHandleInvalid("skip")

# apply it to df
output = assembler.transform(df)



# create string indexer for the status column
indexer = StringIndexer(inputCol='Status', outputCol='StatusIndexer')

output_fixed = indexer.fit(output).transform(output)



#form final dataframe of two cols, features and statusIndexer
df_final = output_fixed.select('features', 'StatusIndexer')

#view format
df_final.show(5, truncate=False)



'''

Splitting the data

train,test
'''

(trainingData, testData) = df_final.randomSplit([0.7, 0.3], seed=42)

#show count
testData.count()

# 1 == Normal
trainingData.filter(trainingData.StatusIndexer == '1').count()

# 0 == Abnormal
trainingData.filter(trainingData.StatusIndexer == '0').count()

# 1 == Normal
testData.filter(testData.StatusIndexer == '1').count()

# 0 == Abnormal
testData.filter(testData.StatusIndexer == '0').count()



'''

Creating the models 


Decision tree, Support vector & Neural network
'''
#classifiers
dtc = DecisionTreeClassifier(labelCol='StatusIndexer', featuresCol='features')
lsvc = LinearSVC(featuresCol='features', labelCol='StatusIndexer', maxIter=10, regParam=0.1)
nnc = MultilayerPerceptronClassifier(labelCol='StatusIndexer', featuresCol='features', layers=[12, 4, 4, 2], seed=123)


# Fit the models to training data
lsvcModel = lsvc.fit(trainingData)
dtc_model = dtc.fit(trainingData)
nncModel = nnc.fit(trainingData)


#now fit the models to the traininf data
dtc_pred = dtc_model.transform(testData)
lsv_pred = lsvcModel.transform(testData)
nnc_pred = nncModel.transform(testData)


# using pyspark evaluator to check performance
acc_eval = MulticlassClassificationEvaluator(labelCol='StatusIndexer', predictionCol='prediction', metricName='accuracy')

#evaluate
dtc_acc = acc_eval.evaluate(dtc_pred)
lsv_acc = acc_eval.evaluate(lsv_pred)
nnc_acc = acc_eval.evaluate(nnc_pred)


#accuracys of models
print(f'Decision Tree Accuracy: {dtc_acc*100} %')
print(f'Support Vector Accuracy: {lsv_acc*100} %')
print(f'Neural Network Accuracy: {nnc_acc*100} %')




'''

Confusion Matrix

'''


# DECISION TREE
tp = dtc_pred[(dtc_pred.StatusIndexer == 1) & (dtc_pred.prediction == 1)].count()
tn = dtc_pred[(dtc_pred.StatusIndexer == 0) & (dtc_pred.prediction == 0)].count()
fp = dtc_pred[(dtc_pred.StatusIndexer == 0) & (dtc_pred.prediction == 1)].count()
fn = dtc_pred[(dtc_pred.StatusIndexer == 1) & (dtc_pred.prediction == 0)].count()

#Accuracy, Error Rate, Specificity & Sensitivity
acc_r = ((tp + tn) / (tp+tn+fp+fn))*100
error_r = ((fp + fn) / (tp+tn+fp+fn))*100
specificity = (tn / (fn+tn))
sensitivity = (tp / (tp+fp))

print(f'dtc Accuracy: {acc_r}%')
print(f'dtc Error rate: {error_r}%')
print(f'dtc Specificity: {specificity}')
print(f'dtc sensitivity: {sensitivity}')


# SUPPORT VECTOR
tp = lsv_pred[(lsv_pred.StatusIndexer == 1) & (lsv_pred.prediction == 1)].count()
tn = lsv_pred[(lsv_pred.StatusIndexer == 0) & (lsv_pred.prediction == 0)].count()
fp = lsv_pred[(lsv_pred.StatusIndexer == 0) & (lsv_pred.prediction == 1)].count()
fn = lsv_pred[(lsv_pred.StatusIndexer == 1) & (lsv_pred.prediction == 0)].count()


#Accuracy, Error Rate, Specificity & Sensitivity
acc_r = ((tp + tn) / (tp+tn+fp+fn))*100
error_r = ((fp + fn) / (tp+tn+fp+fn))*100
specificity = (tn / (fn+tn))
sensitivity = (tp / (tp+fp))

print(f'lsv Accuracy: {acc_r}%')
print(f'lsv Error rate: {error_r}%')
print(f'lsv Specificity: {specificity}')
print(f'lsv sensitivity: {sensitivity}')


# ARTIFICIAL NEURAL NETWORK
tp = nnc_pred[(nnc_pred.StatusIndexer == 1) & (nnc_pred.prediction == 1)].count()
tn = nnc_pred[(nnc_pred.StatusIndexer == 0) & (nnc_pred.prediction == 0)].count()
fp = nnc_pred[(nnc_pred.StatusIndexer == 0) & (nnc_pred.prediction == 1)].count()
fn = nnc_pred[(nnc_pred.StatusIndexer == 1) & (nnc_pred.prediction == 0)].count()


#Accuracy, Error Rate, Specificity & Sensitivity
acc_r = ((tp + tn) / (tp+tn+fp+fn))*100
error_r = ((fp + fn) / (tp+tn+fp+fn))*100
specificity = (tn / (fn+tn))
sensitivity = (tp / (tp+fp))

print(f'ann Accuracy: {acc_r}%')
print(f'ann Error rate: {error_r}%')
print(f'ann Specificity: {specificity}')
print(f'ann sensitivity: {sensitivity}')










'''

Map Reduce

'''

#functions
def Map(x):
    return(x[0], 1)
    
def getMin(x, y):
    if x[0] < y[0]:
        return x
    else:
        return y
    
def getMax(x, y):
    if x[0] > y[0]:
        return x
    else:
        return y

def getMean(x, y):
    return((x[0]+y[0], x[1]+y[1]))

#read in Big nuclear plants dataset / headers not relevant with this task so left alone
datasetBig = spark.read.csv('nuclear_plants_big_dataset.csv', header=True, inferSchema=True).drop('Status')

#lists to store the minimum, maximum and mean
minList = []
maxList = []
meanList = []

# map and reduce
for i in datasetBig.columns:
    datasetBigRdd = (datasetBig.select(i)).rdd
    mapped = datasetBigRdd.map(Map)
    
    findMin = mapped.reduce(getMin)[0]
    minList.append(findMin)
    
    findMax = mapped.reduce(getMax)[0]
    maxList.append(findMax)
    
    ####  Map and Reduce mean // Not working
    #findMean = mapped.reduce(getMean)[0]
    #meanList.append(findMean[0] / findMean[1])


## printing the results

print('Min values: ')
for i in minList:
    
    print(str(i))
       
     

print('\nMax values: ')
for i in maxList:

    print(str(i))


    
    ## Print mean value, but not working
#print('Mean values: ')
#for i in meanList:
#    print(str(i))