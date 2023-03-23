# School budgets

A data science solo project of machine learning for the [Drivendata's competition of US school budget classification](https://www.drivendata.org/competitions/46/box-plots-for-education-reboot/).

In 2022 I hold the [2th place](https://www.drivendata.org/competitions/46/box-plots-for-education-reboot/leaderboard/) of thousands competitors.

It's been a great learning opportunity to solve such a real business problem: to develop a predictive model able to classify school budget detail lines otherwise partially done by specialized consultants which are costly and not scalable US wide.

Although I have many years as computer scientist which helped me a lot, I've been developing data science skills since 2016 from from the sources like [Coursera's Johns Hopkins data science specialization](https://www.coursera.org/specializations/jhu-data-science), [Coursera Stanfords's machine learning course](https://www.coursera.org/learn/machine-learning), [Datacamp's data scientist with Python career track](https://www.datacamp.com/tracks/data-scientist-with-python), [FIOCRUZ Masters's data science classes](https://www.icict.fiocruz.br/), [Stack Exchange](https://stackoverflow.com/), [GitHub](https://github.com/), [DrivenData](https://www.drivendata.org/) and [Kaggle competitions](https://www.kaggle.com/).

Techniques employed in this project:

* Multi-processing
* Exploratory Data Analysis
* Stratified sampling optimization
* Supervised Machine Learning Classification
* Natural Language Processing (NLP)
* Sparse matrices and hashing trick computation
* Learning and validation curves
* Hyperparameter search

The model is developed using Python Jupyter Notebooks for exploratory data analysis (EDA), model development, stratified sampling optimizations and learning curves.

The actual challenge is to accurately classified hundreds of thousands of budget lines from US schools, based on two numeric columns (headcount and money), millions of non-standardized budget text descriptions and a high percentage (29%) of missing data.

The performance metric used is multi-multiclass log loss, scoring predictions of 104 data classes under 9 data labels.  That metric is used to measure the performance of a multiclass multilabel classification optimization problems.

The development environment used deploys Intel's i7, 16 threads, 128GB RAM, last generation 2TB SSD, Geforce RTX3060, Ubuntu Linux, Anaconda, PyCharm IDE, Python 3.9 and scikit-learn.

I've been researching and developing a Python package called [mcr](https://github.com/mauriciocramos/mcr) which is helping me a lot on Machine Learning projects.

Any comments or interest to share & learn in this project are very welcome.