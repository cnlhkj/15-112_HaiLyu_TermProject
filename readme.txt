
Introduction

Old Driver classifier is a small website designed for processing small dataset. It has three main funcionalities:

Auto-classification: Find the best model among Naive Bayes, Decision Tree, Random Forest and Support Vector Machine for your dataset.
Exploration: Help you explore your dataset by proper visualization
DIY: Tune parameters for the model by yourself.


Hope this website can be an old driver who could help you speed up the process of data analytics!

The following part is an instruction of how to make the website run locally on your own computer.

0. Python Version: >= Python 3.4.4

1. Packages and software required:

Packages:
numpy
scipy
pandas
sklearn
matplotlib
seaborn
django
django-registration-redux\

Software:

Graphviz



2. Packages installation instructions:
If you are using Mac, just use pip to install the packages. You can use the codes below:

pip3 install numpy
pip3 install scipy
pip3 install pandas
pip3 install sklearn
pip3 install matplotlib
pip3 install seaborn
pip3 install django
pip3 install django-registration-redux


Notice that you need to use pip to install those packages if you are using Python 2. But there is no guarantee that my codes will work for Python 2.

If you are using Windows or Linux, please check the packages's official websites.


Also, please install the software Graphviz. The download link is shown as below:
http://www.graphviz.org/Download.php

ATTENTION: the order of installing those packages matters, because some packages are built on other packages. Please install them one by one.


3. How to make the website work:

Change your directory into "oldDriverClassifier_codes" directory.

Then run following commands in your terminal:

$ python manage.py runserver

Then go to http://127.0.0.1:8000/

Get registered and enjoy!

Note if you want to use interactive Python to play around with my codes, use the following command to invoke Python shell:

python manage.py shell

4. You will be informed if the website goes online.

