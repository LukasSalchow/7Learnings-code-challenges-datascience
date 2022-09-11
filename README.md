## 7Learnings coding challange. 

Note that is is a very ruff analysis due to time containts. The original task can be found here https://github.com/7Learnings/code-challenges/tree/master/datascience and https://github.com/LukasSalchow/7Learnings-code-challenges-datascience/blob/main/Coding%20Challenge.ipynb

## Some things left one due to time constrains

More data Cleaning:
- Scatter plots, time vs columns
- Outlier detection
- Statistical tests if columns have correct distribution

More Analysis:
- Plot temperature diagrams and compare them to known ones
- Look more at ppscore of time dependencies
- Find predictive patterns for snow
- Read climate science

Better Models:
- Try different models/different kernels
- Use an ensamble of the models that have performed the best
- Use a proper time series model e.g. ARIMA, recurrent neural network / LSTM
- Don’t under sample to deal with imbalance rather try: 
    - Oversampling snow days
    - Data augmentation of snow days
    - Use all data with different metric e.g. a weighted one or F1 Metric
    - auroc (area under roc curve/likelihood of model distinguishing observations from two classes)
