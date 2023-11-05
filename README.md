# CDC Diabetes Health Indicators
This project is a midterm project created by Nadia Paz as part of __[Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master)__

<svg xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="100" height="100" viewBox="0 0 48 48">
<path fill="#0277BD" d="M24.047,5c-1.555,0.005-2.633,0.142-3.936,0.367c-3.848,0.67-4.549,2.077-4.549,4.67V14h9v2H15.22h-4.35c-2.636,0-4.943,1.242-5.674,4.219c-0.826,3.417-0.863,5.557,0,9.125C5.851,32.005,7.294,34,9.931,34h3.632v-5.104c0-2.966,2.686-5.896,5.764-5.896h7.236c2.523,0,5-1.862,5-4.377v-8.586c0-2.439-1.759-4.263-4.218-4.672C27.406,5.359,25.589,4.994,24.047,5z M19.063,9c0.821,0,1.5,0.677,1.5,1.502c0,0.833-0.679,1.498-1.5,1.498c-0.837,0-1.5-0.664-1.5-1.498C17.563,9.68,18.226,9,19.063,9z"></path><path fill="#FFC107" d="M23.078,43c1.555-0.005,2.633-0.142,3.936-0.367c3.848-0.67,4.549-2.077,4.549-4.67V34h-9v-2h9.343h4.35c2.636,0,4.943-1.242,5.674-4.219c0.826-3.417,0.863-5.557,0-9.125C41.274,15.995,39.831,14,37.194,14h-3.632v5.104c0,2.966-2.686,5.896-5.764,5.896h-7.236c-2.523,0-5,1.862-5,4.377v8.586c0,2.439,1.759,4.263,4.218,4.672C19.719,42.641,21.536,43.006,23.078,43z M28.063,39c-0.821,0-1.5-0.677-1.5-1.502c0-0.833,0.679-1.498,1.5-1.498c0.837,0,1.5,0.664,1.5,1.498C29.563,38.32,28.899,39,28.063,39z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="100" height="100" viewBox="0 0 48 48">
<rect width="5" height="10" x="18" y="4" fill="#1a237e"></rect><rect width="5" height="10" x="18" y="25" fill="#1a237e"></rect><rect width="5" height="5" x="18" y="17" fill="#fbc02d"></rect><rect width="5" height="33" x="10" y="13" fill="#1a237e"></rect><rect width="5" height="10" x="26" y="33" fill="#1a237e"></rect><rect width="5" height="10" x="26" y="12" fill="#1a237e"></rect><rect width="5" height="5" x="26" y="25" fill="#ff4081"></rect><rect width="5" height="33" x="34" y="2" fill="#1a237e"></rect></svg> <svg xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="100" height="100" viewBox="0 0 48 48">
<polygon fill="#00acc1" points="21.196,12.276 14.392,8.842 6.922,12.569 13.912,16.078"></polygon><polygon fill="#00acc1" points="24.317,13.85 31.451,17.453 24.049,21.169 17.049,17.654"></polygon><polygon fill="#00acc1" points="33.846,8.893 41.176,12.569 34.619,15.86 27.47,12.254"></polygon><polygon fill="#00acc1" points="30.69,7.31 24.091,4 17.564,7.258 24.364,10.687"></polygon><polygon fill="#00acc1" points="25.532,35.725 25.532,44.73 33.525,40.74 33.518,31.732"></polygon><polygon fill="#00acc1" points="33.514,28.587 33.505,19.674 25.532,23.637 25.532,32.554"></polygon><polygon fill="#00acc1" points="43.111,26.918 43.111,35.957 36.292,39.359 36.287,30.361"></polygon><polygon fill="#00acc1" points="43.111,23.756 43.111,14.898 36.279,18.294 36.285,27.225"></polygon><path fill="#448aff" d="M22.71,23.637l-5.384-2.708v11.699c0,0-6.586-14.012-7.195-15.27 c-0.079-0.163-0.401-0.341-0.484-0.385C8.46,16.353,5,14.601,5,14.601v20.676l4.787,2.566V27.031c0,0,6.515,12.52,6.582,12.657 s0.718,1.455,1.418,1.919c0.929,0.618,4.919,3.016,4.919,3.016L22.71,23.637z"></path>
</svg><img src="data/matplotlib.png" width="96" height="96"> <img src="data/seaborn.svg" width="96" height="96"> <img src="data/sklearn.png"><svg xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="100" height="100" viewBox="0 0 48 48"><path fill="#2395ec" d="M47.527,19.847c-0.13-0.102-1.345-1.007-3.908-1.007c-0.677,0.003-1.352,0.06-2.019,0.171 c-0.496-3.354-3.219-4.93-3.345-5.003l-0.688-0.392l-0.453,0.644c-0.567,0.866-1.068,1.76-1.311,2.763 c-0.459,1.915-0.18,3.713,0.806,5.25C35.417,22.928,33.386,22.986,33,23H1.582c-0.826,0.001-1.496,0.66-1.501,1.474 c-0.037,2.733,0.353,5.553,1.306,8.119c1.089,2.818,2.71,4.894,4.818,6.164C8.567,40.184,12.405,41,16.756,41 c1.965,0.006,3.927-0.169,5.859-0.524c2.686-0.487,5.271-1.413,7.647-2.74c1.958-1.119,3.72-2.542,5.219-4.215 c2.505-2.798,3.997-5.913,5.107-8.682c0.149,0,0.298,0,0.442,0c2.743,0,4.429-1.083,5.359-1.99 c0.618-0.579,1.101-1.284,1.414-2.065L48,20.216L47.527,19.847z"></path><path fill="#2395ec" d="M8,22H5c-0.552,0-1-0.448-1-1v-3c0-0.552,0.448-1,1-1h3c0.552,0,1,0.448,1,1v3 C9,21.552,8.552,22,8,22z"></path><path fill="#2395ec" d="M14,22h-3c-0.552,0-1-0.448-1-1v-3c0-0.552,0.448-1,1-1h3c0.552,0,1,0.448,1,1v3 C15,21.552,14.552,22,14,22z"></path><path fill="#2395ec" d="M20,22h-3c-0.552,0-1-0.448-1-1v-3c0-0.552,0.448-1,1-1h3c0.552,0,1,0.448,1,1v3 C21,21.552,20.552,22,20,22z"></path><path fill="#2395ec" d="M26,22h-3c-0.552,0-1-0.448-1-1v-3c0-0.552,0.448-1,1-1h3c0.552,0,1,0.448,1,1v3 C27,21.552,26.552,22,26,22z"></path><path fill="#2395ec" d="M14,16h-3c-0.552,0-1-0.448-1-1v-3c0-0.552,0.448-1,1-1h3c0.552,0,1,0.448,1,1v3 C15,15.552,14.552,16,14,16z"></path><path fill="#2395ec" d="M20,16h-3c-0.552,0-1-0.448-1-1v-3c0-0.552,0.448-1,1-1h3c0.552,0,1,0.448,1,1v3 C21,15.552,20.552,16,20,16z"></path><path fill="#2395ec" d="M26,16h-3c-0.552,0-1-0.448-1-1v-3c0-0.552,0.448-1,1-1h3c0.552,0,1,0.448,1,1v3 C27,15.552,26.552,16,26,16z"></path><path fill="#2395ec" d="M26,10h-3c-0.552,0-1-0.448-1-1V6c0-0.552,0.448-1,1-1h3c0.552,0,1,0.448,1,1v3 C27,9.552,26.552,10,26,10z"></path><path fill="#2395ec" d="M32,22h-3c-0.552,0-1-0.448-1-1v-3c0-0.552,0.448-1,1-1h3c0.552,0,1,0.448,1,1v3 C33,21.552,32.552,22,32,22z"></path></svg>  <svg xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="100" height="100" viewBox="0 0 48 48"><path fill="#252f3e" d="M13.527,21.529c0,0.597,0.064,1.08,0.176,1.435c0.128,0.355,0.287,0.742,0.511,1.161 c0.08,0.129,0.112,0.258,0.112,0.371c0,0.161-0.096,0.322-0.303,0.484l-1.006,0.677c-0.144,0.097-0.287,0.145-0.415,0.145 c-0.16,0-0.319-0.081-0.479-0.226c-0.224-0.242-0.415-0.5-0.575-0.758c-0.16-0.274-0.319-0.58-0.495-0.951 c-1.245,1.483-2.81,2.225-4.694,2.225c-1.341,0-2.411-0.387-3.193-1.161s-1.181-1.806-1.181-3.096c0-1.37,0.479-2.483,1.453-3.321 s2.267-1.258,3.911-1.258c0.543,0,1.102,0.048,1.692,0.129s1.197,0.21,1.836,0.355v-1.177c0-1.225-0.255-2.08-0.75-2.58 c-0.511-0.5-1.373-0.742-2.602-0.742c-0.559,0-1.133,0.064-1.724,0.21c-0.591,0.145-1.165,0.322-1.724,0.548 c-0.255,0.113-0.447,0.177-0.559,0.21c-0.112,0.032-0.192,0.048-0.255,0.048c-0.224,0-0.335-0.161-0.335-0.5v-0.79 c0-0.258,0.032-0.451,0.112-0.564c0.08-0.113,0.224-0.226,0.447-0.339c0.559-0.29,1.229-0.532,2.012-0.726 c0.782-0.21,1.612-0.306,2.49-0.306c1.9,0,3.289,0.435,4.183,1.306c0.878,0.871,1.325,2.193,1.325,3.966v5.224H13.527z M7.045,23.979c0.527,0,1.07-0.097,1.644-0.29c0.575-0.193,1.086-0.548,1.517-1.032c0.255-0.306,0.447-0.645,0.543-1.032 c0.096-0.387,0.16-0.855,0.16-1.403v-0.677c-0.463-0.113-0.958-0.21-1.469-0.274c-0.511-0.064-1.006-0.097-1.501-0.097 c-1.07,0-1.852,0.21-2.379,0.645s-0.782,1.048-0.782,1.854c0,0.758,0.192,1.322,0.591,1.709 C5.752,23.786,6.311,23.979,7.045,23.979z M19.865,25.721c-0.287,0-0.479-0.048-0.607-0.161c-0.128-0.097-0.239-0.322-0.335-0.629 l-3.752-12.463c-0.096-0.322-0.144-0.532-0.144-0.645c0-0.258,0.128-0.403,0.383-0.403h1.565c0.303,0,0.511,0.048,0.623,0.161 c0.128,0.097,0.223,0.322,0.319,0.629l2.682,10.674l2.49-10.674c0.08-0.322,0.176-0.532,0.303-0.629 c0.128-0.097,0.351-0.161,0.639-0.161h1.277c0.303,0,0.511,0.048,0.639,0.161c0.128,0.097,0.239,0.322,0.303,0.629l2.522,10.803 l2.762-10.803c0.096-0.322,0.208-0.532,0.319-0.629c0.128-0.097,0.335-0.161,0.623-0.161h1.485c0.255,0,0.399,0.129,0.399,0.403 c0,0.081-0.016,0.161-0.032,0.258s-0.048,0.226-0.112,0.403l-3.847,12.463c-0.096,0.322-0.208,0.532-0.335,0.629 s-0.335,0.161-0.607,0.161h-1.373c-0.303,0-0.511-0.048-0.639-0.161c-0.128-0.113-0.239-0.322-0.303-0.645l-2.474-10.4 L22.18,24.915c-0.08,0.322-0.176,0.532-0.303,0.645c-0.128,0.113-0.351,0.161-0.639,0.161H19.865z M40.379,26.156 c-0.83,0-1.66-0.097-2.458-0.29c-0.798-0.193-1.421-0.403-1.836-0.645c-0.255-0.145-0.431-0.306-0.495-0.451 c-0.064-0.145-0.096-0.306-0.096-0.451v-0.822c0-0.339,0.128-0.5,0.367-0.5c0.096,0,0.192,0.016,0.287,0.048 c0.096,0.032,0.239,0.097,0.399,0.161c0.543,0.242,1.133,0.435,1.756,0.564c0.639,0.129,1.261,0.193,1.9,0.193 c1.006,0,1.788-0.177,2.331-0.532c0.543-0.355,0.83-0.871,0.83-1.532c0-0.451-0.144-0.822-0.431-1.129 c-0.287-0.306-0.83-0.58-1.612-0.838l-2.315-0.726c-1.165-0.371-2.027-0.919-2.554-1.645c-0.527-0.709-0.798-1.499-0.798-2.338 c0-0.677,0.144-1.274,0.431-1.79s0.671-0.967,1.149-1.322c0.479-0.371,1.022-0.645,1.66-0.838C39.533,11.081,40.203,11,40.906,11 c0.351,0,0.718,0.016,1.07,0.064c0.367,0.048,0.702,0.113,1.038,0.177c0.319,0.081,0.623,0.161,0.91,0.258s0.511,0.193,0.671,0.29 c0.224,0.129,0.383,0.258,0.479,0.403c0.096,0.129,0.144,0.306,0.144,0.532v0.758c0,0.339-0.128,0.516-0.367,0.516 c-0.128,0-0.335-0.064-0.607-0.193c-0.91-0.419-1.932-0.629-3.065-0.629c-0.91,0-1.628,0.145-2.123,0.451 c-0.495,0.306-0.75,0.774-0.75,1.435c0,0.451,0.16,0.838,0.479,1.145c0.319,0.306,0.91,0.613,1.756,0.887l2.267,0.726 c1.149,0.371,1.98,0.887,2.474,1.548s0.734,1.419,0.734,2.257c0,0.693-0.144,1.322-0.415,1.87 c-0.287,0.548-0.671,1.032-1.165,1.419c-0.495,0.403-1.086,0.693-1.772,0.903C41.943,26.043,41.193,26.156,40.379,26.156z"></path><path fill="#f90" d="M43.396,33.992c-5.252,3.918-12.883,5.998-19.445,5.998c-9.195,0-17.481-3.434-23.739-9.142 c-0.495-0.451-0.048-1.064,0.543-0.709c6.769,3.966,15.118,6.369,23.755,6.369c5.827,0,12.229-1.225,18.119-3.741 C43.508,32.364,44.258,33.347,43.396,33.992z M45.583,31.477c-0.671-0.871-4.438-0.419-6.146-0.21 c-0.511,0.064-0.591-0.387-0.128-0.726c3.001-2.128,7.934-1.516,8.509-0.806c0.575,0.726-0.16,5.708-2.969,8.094 c-0.431,0.371-0.846,0.177-0.655-0.306C44.833,35.927,46.254,32.331,45.583,31.477z"></path></svg> 

<h2 style="color:#777;">1. Project's description</h2>
<details><summary><i>Expand</i></summary>
Diabetes and prediabetes are national epidemics impacting more than 133 million Americans, and diabetes is one of the fastest growing chronic diseases in the world (<a href="https://diabetes.org/about-us/annual-reports">source</a>). Type 2 diabetes is largely preventable by taking several simple steps: keeping weight under control, exercising more, eating a healthy diet, and not smoking (<a href="https://www.hsph.harvard.edu/nutritionsource/disease-prevention/diabetes-prevention/preventing-diabetes-full-story">source</a>). This project involves CDC healthcare statistics and information gathered from lifestyle surveys. The data I am working with includes basic demographics and health questionnaire responses from participants, where they answer questions about their activities, lifestyle, health indicators, and their diabetes condition (diabetes, pre-diabetes, or healthy). The information has been adjusted for binary classification and has only two possible outcomes: diabetes or healthy. The objective of this project is to create a machine learning classification model that is capable of predicting the probability of contracting diabetes based on health and lifestyle data. The purpose of this project is to raise awareness about diabetes and encourage people to adopt a healthier lifestyle.
</details>

<h2 style="color:#777;">2. Data Source and Aquisition</h2>
<details><summary><i>Expand</i></summary>

I downloaded the data for the project from the <a href="https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators">UC Irvine Machine Learning Repository</a>. If you're interested in obtaining the same data, you can access it in the "data" folder of this project or visit the UC Irvine website and follow the instructions outlined in the "Import in Python" section. After getting `X` an `y` variables, I merged the data into a data frame and saved it as a `csv` file with following code:

```python
import os
# merge data
df = pd.concat([X, y], axis = 1)
# save to csv
df.to_csv('diabetis_data.csv', index_label=False)
```

The same data is also available on <a href="https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv">Kaggle</a>, but be aware that the column names and order differ from those used in the project.


Data description can be find on <a href="https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv">Kaggle</a> or at Behavioral Risk Factor Surveillance System 2015 <a href="https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf">Codebook Report</a>

### Data dictionary

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

### Data manipulations
The original data consists from 253,680 rows and 22 columns. I dropped 24,206 duplicated rows, that leaves us with 229,474 rows. 


</details>

<h2 style="color:#777;">3. Download the project</h2>

<details><summary><i>Expand</i></summary>

You can download it from this GitHub repository by selecting `Code` -> `Download ZIP`, or run the command `git clone git@github.com:nadia-paz/cdc_diabetis.git`
`cd cdc_diabetis` to move to the project's directory. <br>
[`git clone`](https://www.atlassian.com/git/tutorials/setting-up-a-repository/git-clone)

</details>

<h2 style="color:#777;">4. Virtual environtments</h2>
<details><summary><i>Expand</i></summary>

#### Anaconda

The project is made using Python 3.9.18 on Anaconda. To create the same virtual environment with Anaconda please refer to the file `environment.yml`. Install Anaconda or Mamba if you don't have it yet and run the following command in your terminal from the project's directory:

```bash
conda env create -f environment.yml
```
The name of the environment already is specified in the file. After installing the environment activate it with the command:

```bash
conda activate diabetes_project
```
and start `jupyter notebook`

To deactivate the environment: `conda deactivate`

#### `venv`
If you don't have Anaconda or don't want to use it, you can install required dependencies using Python's `venv`. They are located in the `venv_requirements.txt` file. 

__Step 1: Install Python 3.9__

Check your Python's version in your terminal: `python --version` or `python3 --version`. If it is different from the Python 3.9.*, install Python 3.9 on your computer according with your operation system instructions. For Linux `sudo apt-get install python3.9`, for Mac `brew install python@3.9`, for Windows manually download and install the required Python's version. 

__Step 2: Locate the path of your Python3.9__

Run in your terminal `which python3.9`. Copy the output. It is your `path_to_python`

__Step 3: Create a virtual environment__
1. In your terminal move to the projects folder `cd <path_to_the_project>`.  
2. Create the environment. In your terminal run the command

```bash
<path_to_python> -m vevn <env_name>
```
__Step 4: Activate the virtual environment__

In terminal run:
```bash
source <env_name>/bin/activate
```
__Step 5: Install dependendencies__

Make sure that you are in the project's directory and you have the `venv_requirements.txt` file in it. Run the following command in the terminal:

```bash
python -m pip install -r venv_requirements.txt
```
Now you can use the project. To deactivate the virtual environment simpy run `deactivate` in the terminal.

#### Confirm virual environment from `jupyter notebook`
In the code cell run:
```python
import os
print(os.system("which python"))
print(os.system("python --version"))
```
You should see the name of your environment in the output. If you don't, confirm the installation of the environment to the iPython Kernell. In the terminal window run:
```bash
ipython kernel install --user --name=<env_name>
```
</details>

<h2 style="color:#777;">4. Project's files</h2>
<details><summary><i>Expand</i></summary>

```
├── data
│   └── diabetis_data.csv
│   ├── matplotlib.png
│   ├── seaborn.svg
│   ├── sklearn.png
├── deployment
│   ├── Dockerfile
│   ├── Pipfile
│   ├── Pipfile.lock
│   ├── encoder.bin
│   ├── model.bin
│   ├── predict.py
│   ├── test.py
│   └── test_aws.py
├── src
│   ├── data_prep.py
│   ├── explore.py
│   ├── model.py
│   └── transform.py
├── environment.yml
├── notebook.ipynb
├── train.py
├── use_model.py
└── venv_requirements.txt
├── README.md
```
Directories:
- `data`: contains `*.csv` file with the data
- `deployment`: contains binary files with the model and OneHotEncoder that was fit on the `train` set, files to use the model on Docker, and `test_aws.py` that can be copied anywhere and is used to send requests to the web-zpplication hosted on AWS.
- `src` contains files that assist data preparation `data_prep.py`, exploration `explore.py`, model tuning `model.py`, and transformation of the single patient from the dictionary into `numpy` array ready to use in the model.

The project's main notebook with step by step exploration, tuning and saving the model is `notebook.ipynb`. The script needed to build the model is in `train.py` file. It contains the code from the notebook and files located in `src` directory, that was used for the model development.

</details>

<h2 style="color:#777;">5. How to use the model</h2>

You can use the model in the virtual environment of the project, on Docker or send a request to the AWS web service (temporary option and will be deprecated soon). Every file that is used for testing  contains a dictionary `patient` with the basic information. If you decide to change that information, please, refer to the Data Dictionary prior to making changes (part 2. __Data Source and Aquisition__ of this `Readme` file).


### Virtual environment of the project
On your terminal move to the project's directory, activate the virtual environment and run the command `python use_model.py` or `python3 use_model.py`.

### Docker
1. Download and install Docker if you don't have it on your machine.
2. The virtual environment and deployment files are located in the directory `deployment`.
2. **Build Docker image**
    - On your termnial move to the directory "deployment".
    - Run the following command:

        `docker build -t diabetes-project .`

        This will build the Docker image on your machine.
4. Next, run Docker image. 

    `docker run --rm -p 2912:2912 diabetes-project`

    This command launches the container with the diabetes prediction model and listens for your requests on localhost port `2912`. To send this request you have to open a new terminal window, move to the deployment directory and run the script `test.py`.

### Web application
This model is deployed as a web application on AWS Elastic Beanstalk. To use it run `test_aws.py` file located in deployment folder.

<details><summary><i>Expand</i></summary>

Icons for Technologies in the Readme file are from:
- [Seaborn](https://seaborn.pydata.org/citing.html)
- __Matplotlib__ [cleanpng.com](https://www.cleanpng.com/)
- __Sci-Kit Learn__ [pngegg.com](https://www.pngegg.com/)
- Everything else from [Icons8](https://icons8.com/icons)

</details>