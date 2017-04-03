# Approach 

## Initialization
I started out by trying a basic xgboost model using the given features and filling the missing values with -1. I generally start with xgboost because of its speed and good scores. I had removed the ID and timestamp featuers. 

## Cross Validation
To set up a quick cross validation, I randomly sampled out 10% of the dataset and set that up as the eval data. I had planned to write for timestamp based partitioning later. But the initial eval scores for this setup were similar to the ones I got on the public leaderboard, so I persisted with this setup.  

## Feature Engineering
On plotting the feature importances using the default set of features, I realized that the MA features were not contributing much. Also, to me using the absolute values of these features was not intutive. Removing these gave me an improvement in the eval score as well as the public leaderboard score. Then I removed the volume traded feature because it was also having a low contribution and removing it gave me an improvement in both eval and public lb. Later, i created 3 new features:
- difference between three day moving average and five day moving average 
- difference between five day moving average and ten day moving average 
- difference between positive directional movemeent and negative directional movement

I added these features one by one and saw an improvement in both the eval and public lb scores. 

I tried creating a feature for differnce between three day moving average of nth day minus the three day moving average of (n-1)th day. This gave me improvement in eval dataset, but not on the public lb. Possibily this had overfit the data, so I removed this feature. 

## Parameter Tuning 

### max depth
I usually start with shallow trees (max depth 3). I prefer to use shallow trees because they dont tend to overfit. I tried increasing the max depth to 4 and 5, but that made the scores worse for public lb. So I stuck to using max depth 3. 

### min_child_weight
Initially, I set the min_child_weight to 1000 because of the high number of data points. Later I moved it to 1500 and 500 and saw that 500 gave me a better score. Decreasing further to 300 didnt help so I stuck with 500. 

### Learning Rate, num_rounds and early stopping
I set up the early stopping parameter to 50, i.e. if the eval score doesnt improve in 50 rounds, stop training further. The learning rate was initially set to 0.05 and num rounds were initially set to 1500. But this was very slow and the score was improving even after 1500 rounds. So I changed the learning rate to 0.2 and reduced the num rounds to 800. This gave me stopping near the 600th round and quicker training as a result. 

Well, thats it, I did not have the time to try ensemble models which I believe could have improved the score further. 

# Running the code 
1. Keep all the files(python script, train.csv and test.csv) in the same directory and set the working directory to that directory. 
2. Run the script by command: python try1.py. 
3. The submission is saved as submission_xgb.csv.

## Packages Used
1. scikit-learn 0.18 http://scikit-learn.org/stable/
   <br> sudo pip install scikit-learn
2. pandas 0.18.1 http://pandas.pydata.org/ <br> sudo pip install pandas
3. numpy 1.11.2 http://www.numpy.org/ <br> sudo pip install numpy
4. xgboost 0.6 https://github.com/dmlc/xgboost/tree/master/python-package 

## Dependencies for packages 
1. scipy 0.18.0 https://www.scipy.org/ <br> sudo pip install scipy
2. Theano 0.8.2 http://deeplearning.net/software/theano/install.html#install <br> sudo pip install Theano
3. pyyaml 3.11 https://pypi.python.org/pypi/PyYAML/3.12 <br> sudo pip install pyyaml
4. gcc 4.8.4 <br> sudo apt-get install build-essential

