## Big Data Analysis of Pressurized Nuclear Reactors

#### using Python and PySpark



This project aimed to create code which could sort through and make assumptions on large amounts of data based on Pressurized Nuclear Reactor 
Data provided by Siemens Energy.

The Dataset contains 3 different types of Sensors which are extracted from real world Pressurized Nuclear Reactors.
The data contains 4 of each sensor meaning there are 12 overall pieces of sensor data.

The 3 Sensors are: **Power Range sensor**, **Pressure sensor** and **Vibration sensor**.

The Dataset also contains a column called Status. The column indicates whether the Reactor is working Normally or Abnormally.

A Decision Tree, Learning Support Vector and Neural Network were created to predict the Status Column based on the 12 pieces of sensor data.

The Decision Tree returned the highest accuracy of 82.0%, Meaning it can predict the status of the reactor 82 percent of the time. 



The Jupyter Notebook file demostrates different stages in the development of sorting and cleaning the data, extracting useful information 
and making predictions using the data.




[Click here](https://github.com/douglascarrie/Big-Data-Analysis-of-Pressurized-Nuclear-Reactors/blob/master/Big%20Data%20analysis.pdf) to view the Jupyter file.
<br>

---

Structure of dataset (.CSV file):


<img src="https://github.com/douglascarrie/Big-Data-Analysis-of-Pressurized-Nuclear-Reactors/blob/master/Structure.png" alt="Example of structure" width="200"/>
![Example of dataset](https://github.com/douglascarrie/Big-Data-Analysis-of-Pressurized-Nuclear-Reactors/blob/master/Structure-grid.png)

##### Challenges faced in this project: 
- Using MapReduce to gain mean values from dataset 
- Obtaining higher accuracies from models


---
### Installation

You need Python, Spark, Pyspark. To run Jupyter file you will need Jupyter Notebook installed. 
Best to install with Anaconda.

Requires Small dataset and Large dataset.

Problems: 
- The larger dataset is 143mb, which exceeds GitHubs max file size

---

### Report

A report was created (pdf) to explain further each method used, this was done as part of an assessment.

Link to report: [Click here](https://github.com/douglascarrie/Big-Data-Analysis-of-Pressurized-Nuclear-Reactors/blob/master/Big%20Data%20analysis.pdf)


