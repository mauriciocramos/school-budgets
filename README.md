# school-budgets

A data science solo project of machine learning for the [Drivendata's competition of US school budget classification](https://www.drivendata.org/competitions/46/box-plots-for-education-reboot/).

Merchandising: As of July 3rd 2020, I hold the top 11th place of [competition´s leaderboard](https://www.drivendata.org/competitions/46/box-plots-for-education-reboot/leaderboard/) between 1935 competitors.

It's been a great learning opportunity to solve such a real business problem: to develop a predictive model able to classify school budget detail lines otherwise partially done by specialized consultants which are costly and not scalable US wide.

Although I have many years as computer scientist which helped me a lot, I actually developed data science skills in the last 5 years from sources like [Coursera's Johns Hopkins data science specialization](https://www.coursera.org/specializations/jhu-data-science), [Coursera Stanfords's machine learning course](https://www.coursera.org/learn/machine-learning), [Datacamp's data scientist with Python career track](https://www.datacamp.com/tracks/data-scientist-with-python), [FIOCRUZ Masters's data science classes](https://www.icict.fiocruz.br/), [Stack Exchange](https://stackoverflow.com/), [GitHub](https://github.com/) and [Kaggle competitions](https://www.kaggle.com/).

Techniques employed in this project:

* Exploratory Data Analysis
* Stratified sampling optimization
* Supervised Machine Learning Classification
* Natural Languge Processing (NLP)
* Sparse matrices and hashing trick computation
* Hyperparameter optimization through Learning and Cross-Validation Curves

The model is developed using Python Jupyter Notebooks for [exploratory data analysis](https://github.com/mauriciocramos/school-budgets/blob/master/notebooks/1-Analysis.ipynb), [stratified sampling optimization](https://github.com/mauriciocramos/school-budgets/blob/master/notebooks/2-Stratified%20sampling.ipynb) and [supervised machine learning classification pipelines](https://github.com/mauriciocramos/school-budgets/blob/master/notebooks/3-Model%20development.ipynb).

The actual challenge is to accurately classified hundreds of thousands of budget lines from thousands of schools, based on two numeric columns (people and money), millions of non-standardized budget text descriptions, a high percentage (29%) of missing data and quadcore 1.6Ghz notebook equped with 16 gigabytes of RAM memory.

The performance metric used is multi-multiclass log loss, scoring predictions of 104 data classes under 9 data labels.  That metric is used to measure the performance of a multiclass multilabel classification optimization problem.

The development environment used is Anaconda, containing PyCharm IDE, Python 3.7 and Python's scipy and scikit-learn modules.

Next steps:

1. Data resampling
2. Have more hardware
3. Deep Learning

Any comments or interest to share & learn in this project are very welcome.
