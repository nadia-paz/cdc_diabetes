# CDC Diabetes Health Indicators
This project is a midterm project created by Nadia Paz as part of __[Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master)__


<h2 style="color:#777;">1. Project's description</h2>
<details><summary><i>Expand</i></summary>
Diabetes and prediabetes are national epidemics impacting more than 133 million Americans, and diabetes is one of the fastest growing chronic diseases in the world (<a href="https://diabetes.org/about-us/annual-reports">source</a>). Type 2 diabetes is largely preventable by taking several simple steps: keeping weight under control, exercising more, eating a healthy diet, and not smoking (<a href="https://www.hsph.harvard.edu/nutritionsource/disease-prevention/diabetes-prevention/preventing-diabetes-full-story">source</a>). This project involves CDC healthcare statistics and information gathered from lifestyle surveys. The data I am working with includes basic demographics and health questionnaire responses from participants, where they answer questions about their activities, lifestyle, health indicators, and their diabetes condition (diabetes, pre-diabetes, or healthy). The information has been adjusted for binary classification and has only two possible outcomes: diabetes or healthy. The objective of this project is to create a machine learning classification model that is capable of predicting the probability of contracting diabetes based on health and lifestyle data. The purpose of this project is to raise awareness about diabetes and encourage people to adopt a healthier lifestyle.
</details>

<h2 style="color:#777;">2. Data Source and Aquisition</h2>
<details><summary><i>Expand</i></summary>
Data origin

Data description can be find on Kaggle or at Behavioral Risk Factor Surveillance System 2015 <a href="https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf">Codebook Report</a>

| Column name   | Definition  | Numer of Unique Values   |  Data Type       |
| :------------------| :------------------|         :------------------:|  :------------------| 
|`HighBP`|High blood pressure|2| int|
|`HighChol` |High colesterol |2| int|
|`CholCheck` |Cholesterol check whithin the last 5 years |2|int |
|`BMI`|Body mass index. Normal range is 18.5 - 24.9 |continuous variable|int |
|`Smoker`|The respondent smoked at least 100 cigarettes throu his/her life |2|int |
|`Stroke`|Ever had stroke |2|int |
| `HeartDiseaseorAttack`|The history of heart desease or heart attack |2|int |
|`PhysActivity`|Physical activity in past 30 days not including job|2|int |
|`Fruits`|Consume 1 or more fruit per day |2|int |
|`Veggies`|Consume 1 or more vegetables per day  |2|int |
|`HvyAlcoholConsump`|Heavy alcohol consumption.<br>Men: >= 14 drinks per week<br>Women >= 7 drinks per week|2|int |
|`AnyHealthcare`|Healthcare coverage (insurance, medical plans) |2|int |
|`NoDocbcCost` |No doctor beacuse of cost within past 12 month |2|int |
|`GenHlth` |Would you say that in general your health is: scale 1-5 <br>`1` = excellent <br>`2` = very good <br>`3` = good <br>`4` = fair <br>`5` = poor |5|int |
|`MentHlth` |Days of poor mental health per month (within past 30 days?) |31|int |
|`PhysHlth` |Physical illness or injury whithin past 30 days |31|int |
|`DiffWalk` |Serious difficulty walking or climbing stairs |2|int |
|`Sex` |How many times per week do you have sex? (Kidding)<br>Gender: 0- female, 1 - male |2|int |
|`Age` |Respondent's age by category<br>`1`: 18-24 <br>`2`: 25-29 <br>`3`: 30-34 <br>`4`: 35-39 <br>`5`: 40-44  <br>`6`: 45-49 <br>`7`: 50-54 <br>`8`: 55-59  <br>`9`: 60-64  <br>`10`: 65-69 <br>`11`: 70-74 <br>`12`: 75-80 <br>`13`: 80 years and older|13|int |
|`Education` |Education level by category:<br>`1`, `2`, `3`: didn't graduate from high school <br>`4`: graduated from high school <br>`5`: attended college <br>`6`: graduated from college |6|int |
|`Income` |Income level by category:<br>`1`: less than $10,000 <br>`2`: more than $10,000 less than $15,000 <br>`3`: more than $15,000 less than $20,000 <br>`4`: more than $20,000 less than $25,000 <br>`5`:  more than $25,000 less than $35,000 <br>`6`:  more than $35,000 less than $50,000<br>`7`:  more than $50,000 less than $75,000 <br>`8`: more than $75,000 |8|int |
|**Target Variable**
||
|`Diabetes_binary` |The respondent has diabetis |2|int |

</details>


<h2 style="color:#777;">3. Creating a virtual environtment</h2>

<h2 style="color:#777;">4. </h2>

