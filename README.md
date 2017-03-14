Regression Case Study
======================

In today's exercise you'll get a chance to try some of what you've learned about
supervised learning on a real-world problem.

The goal of the contest is to predict the sale price of a particular piece of
heavy equipment at auction based on it's usage, equipment type, and configuration.
The data is sourced from auction result postings and includes information on usage
and equipment configurations.

Evaluation
======================
The evaluation of your model will be based on Root Mean Squared Log Error. Which
is computed as follows:

![](images/rmsle.png)

where *p<sub>i</sub>* are the predicted values and *a<sub>i</sub>* are the target
values.

See the code in score_model.py for details.

Setup
======================
Run
`pip install git+https://github.com/zipfian/performotron.git`

Data
======================
The data for this case study are in `./data`. Although there are both training and testing data sets,
the testing data set will only be utilized to evaluate your final model performance.  In other words,
you should use cross-validation on the training data set to identify potential models, then score those models on
the test data.

In order to score your model, you will need to
output your predictions in the format specified in `data/median_benchmark.csv`. Then
you can submit your solution for evaluation using the command:

    python score_model.py data/your_predictions.csv

Note that this will announce your score on Slack to everybody else, but feel free to submit early to make sure you have a working model.

Important Tips
=========================

1. This data is quite messy. Try to use your judgement about where your cleaning efforts will yield the most results and focus there first.
2. Remember any transformations you apply to the training data will also have to be applied to the testing data, so plan accordingly.
 * It's possible some columns in the test data will take on values not seen in the training data. Plan accordingly.
3. Use your intuition to think about where the strongest signal about a price is likely to come from. If you weren't fitting a model, but were asked to use this data to predict a price what would you do? Can you combine the model with your intuitive instincts?
4. Start simply. Fit a basic model and make sure you're able to get the submission working then iterate to improve. Try to submit a model--even if you know it has some weaknesses--within the first hour.
