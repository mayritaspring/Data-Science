# Data-Science
![images](https://github.com/mayritaspring/Data-Science/blob/master/figures/data_science.png)

It's all about the skills of data science, containing machine learning, data mining, Python examples and packages.

- Python References
1. Python for Data Analysis: https://ask.hellobi.com/blog/python_shequ/11468

2. Python Code for Practice: https://github.com/wesm/pydata-book

- Machine Learning References
1. Machine Learning Concept: https://www.csie.ntu.edu.tw/~htlin/mooc/

- Popular Machine Learning Algorithm
1. Logistic Regression

2. Classification And Regression Tree (CART) 

3. Random Forest

4. Ensemble

5. Gradient Boosting

	**5-1. 三大Form**

 	a. *Gradient Boosting algorithm* 也稱gradient boosting machine（含learning rate）
 	
	b. *Stochastic Gradient Boosting* with sub-sampling at the row, column and column per split levels.

	c. *Regularized Gradient Boosting* 加入L1和L2之regularization term
	
	**5-2. 簡介**

	a. 該方法可以解決regression和classification的問題
	
	b. Boosting是一個ensemble technique，也就是新模型會修正現有模型所產生的error，而模型將會逐步的調整至無法調整．透過gradient descent algorithm以極小化loss已產生新模型．


	**5-3. Implementation**

	a. XGBoost: 與其它gradient boosting的implementation相比快很多. 且使用Gradient boosting decision tree algorithm (又稱gradient boosting, multiple additive regression trees, stochastic gradient boosting or gradient boosting machines) 

	b. LightGBM: [Link1](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc), [Link2](https://media.readthedocs.org/pdf/testlightgbm/latest/testlightgbm.pdf)

6. Neural Networks

7. SVＭ

- Data Preparation 
1. Cross Validation

	**5-1. 目的** 

	a. Cross Validation的初衷是為了避免依賴某一特定的訓練和驗證資料產生偏差。
舉例來說，10 fold validation切成十份後，會用其中九份訓練，其中一份驗證，總共會做十次．透過這樣人人都可以輪流當訓練和驗證的過程，將十次error取平均當成E(in)就可以讓error的計算不會偏重在每一次建模的error([reference](https://ithelp.ithome.com.tw/articles/10197461))

	b. 接下來，才會接到機器學習當中，調整不同參數同時也做10 fold validation，以計算E(in)做到不同參數所建立模型之比較，進而選出最好的模型([reference](http://blog.fukuball.com/lin-xuan-tian-jiao-shou-ji-qi-xue-xi-ji-shi-machine-learning-foundations-di-shi-wu-jiang-xue-xi-bi-ji/))
