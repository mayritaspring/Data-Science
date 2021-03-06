# Data-Science
![images](https://github.com/mayritaspring/Data-Science/blob/master/figures/data_science.png)

It's all about the skills of data science, containing machine learning, data mining, Python examples and packages.

- Python References
1. Python for Data Analysis: https://ask.hellobi.com/blog/python_shequ/11468

2. Python Code for Practice: https://github.com/wesm/pydata-book

- Machine Learning References
1. Machine Learning Concept: https://www.csie.ntu.edu.tw/~htlin/mooc/

- Popular Topic of Data Science 
1. Logistic Regression

2. Classification And Regression Tree (CART) 
	
	**2-1. 簡介**
	
	a. [Tutorial](http://www.stats.ox.ac.uk/~flaxman/HT17_lecture13.pdf)

3. Random Forest

4. Ensemble

5. Gradient Boosting Machine (GBM)

	**5-1. 核心精神**

	a. 可以解決regression和classification的問題
	
	b. Boosting是一個ensemble technique，會透過序列的方式加入新模型到ensemble中

	c. 運用gradient descent algorithm以極小化loss function的方式求解．透過boosting相對應的模型稱為Gradient Boosting Machine (GBM)

	d. 極小化loss function的方式就是針對loss function做微分得到gradient descent，透過將gradient descent取負號會得到下一次的new base-learner，簡言之就是下一次的新模型會修正現有模型所產生的error，最終模型將會逐步的調整至無法調整．


	**5-2. 架構**
	
	選擇loss function和base-learner是GBM很重要的議題
	
	a. *Loss Function* 連續型的y和類別型的y會有不同的loss function.

		(1) 連續型y: L2 squared loss function, L1 absolute loss function

		(2) 類別型y: Bernoulli loss function, Adaboost loss function(和Adaboost相同的simple exponential loss)

	b. *Base-Learner* 分為以下幾類：

		(1) Linear models (OLS, Ridge regression)

		(2) Smooth models (P-splines, Radial basis functions)

		(3) Decision trees 

		(4) Other models (Markov Random Fields)

	c. *Regularization*

		(1) Subsampling: 需使用到parameter, bag fraction, 介於0~1. 評估每次迭代會抽樣使用多少data去訓練模型

		(2) Shrinkage: 簡言之就是在線性迴歸當中所使用的正規化方式(EX: LASSO, Ridge Regression)，該方法是為了要降低潛在不穩定的回歸係數影響．調整參數為lambda，介於0~1，當lambda越大，也就代表迭代成本越大

		(3) Early stopping: 最適合的boost次數，lambda和boost次數之間有trade-off關係
		
 
	**5-3. 三大Form**

 	a. *Gradient Boosting algorithm* 也稱gradient boosting machine（含learning rate）
 	
	b. *Stochastic Gradient Boosting* with sub-sampling at the row, column and column per split levels.

	c. *Regularized Gradient Boosting* 加入L1和L2之regularization term
	
	
	**5-4. Implementation**

	a. XGBoost: 與其它gradient boosting的implementation相比快很多. 且使用Gradient boosting decision tree algorithm (又稱gradient boosting, multiple additive regression trees, stochastic gradient boosting or gradient boosting machines) 

	b. LightGBM: [Link1](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc), [Link2](https://media.readthedocs.org/pdf/testlightgbm/latest/testlightgbm.pdf)

6. Neural Networks

7. Fatorization Machine
	**7-1. 核心精神**

	a.
 
	**7-2. 求解方式(SGD)**
		
	a. PLA: 選一個點出來，作為錯誤修正的方向．

	b. LR: 每一輪對所有點gradient的貢獻通通算出來．

	c. 以單一輪來看的看，LR 會花較久的時間

	d. SGD: 用隨機挑一個點的方式，取代加總起來取平均，去做偏微分取梯度的動作．即將整體梯度作為這個隨機過程的期望值．簡言之，是用隨機梯度做下降而非真實梯度．原因是當作足夠多次時，真實梯度和隨機梯度會是差不多的．EX:可以用於資料不是一批一批來，而是一筆一筆來的情況，稱online learning

	e. SGD logistic regression 近似於 soft PLA，原因是不像原本PLA只要有錯就更新，而是透過微幅修正的方式進行，錯得多就多更新一點

	f. SGD實務法則: 決定stopping condition，t足夠長．修正幅度設定為0.1為實務常用做法．


	**7-3. epoch**

	a. iteration：表示1次迭代（也叫訓練步驟），每次迭代更新1次網絡結構的參數

	b. batch-size：1次迭代所使用的樣本量

	c. epoch：1個epoch表示過了1遍訓練集中的所有樣本。值得注意的是，在深度學習領域中，常用帶mini-batch的隨機梯度下降算法（Stochastic Gradient Descent，SGD）訓練深層結構，它有一個好處就是並不需要遍歷全部的樣本，當數據量非常大時十分有效。此時，可根據實際問題來定義歷元，例如定義10000次迭代為1個時期，若每次迭代的批量大小設為256，那麼1個歷元相當於過了256萬個訓練樣本。

8. Deep Learning: CNN/RNN(PyTorch)

9. NLP

10. Embedding

	**10-1. Word Embedding**
		
	a. 重點

		(1) 為一種Dimension Reduction的方法

		(2) training data是大量text，generating word vector is unsupervised

		(3) 無法用auto encoder處理得到一樣結果
	b. 類型
		(1) Count-Based:計算text共同出現次數

		(2) Prediction-Based:Architectures分成CBOW和Skip-gram

		EX:中文和英文詞彙是不會在附近的，除非同一個句子裡面同時出現中文和英文

	**10-2. Neighbor Embedding**

	a. 重點


	

- Data Preparation 
1. Cross Validation

	**1-1. 目的** 

	a. Cross Validation的初衷是為了避免依賴某一特定的訓練和驗證資料產生偏差。
舉例來說，10 fold validation切成十份後，會用其中九份訓練，其中一份驗證，總共會做十次．透過這樣人人都可以輪流當訓練和驗證的過程，將十次error取平均當成E(in)就可以讓error的計算不會偏重在每一次建模的error([reference](https://ithelp.ithome.com.tw/articles/10197461))

	b. 接下來，才會接到機器學習當中，調整不同參數同時也做10 fold validation，以計算E(in)做到不同參數所建立模型之比較，進而選出最好的模型([reference](http://blog.fukuball.com/lin-xuan-tian-jiao-shou-ji-qi-xue-xi-ji-shi-machine-learning-foundations-di-shi-wu-jiang-xue-xi-bi-ji/))

2. Variance v.s. Bias

	**2-1. 說明**
	
	Variance為Out sample error， Bias為In sample error

- Feature Engineering

- Feature Selection

- Model Selection
1. 自動調參算法: 常用的有Grid search（網格搜索）、Random search（隨機搜索）、Bayesian Optimization。

	**1-1. Grid Search**

	Manual grid search works for low-dimensional problems, but does not scale well to higher dimensions.

	**1-2. Bayesian Optimization**

	a. 適用情形: Hyperparameter很多或是computationally expensive
	
	b. Gaussisan Process ([Link](https://zhuanlan.zhihu.com/p/27555501))

		b-1. Exploration(會往variance大的地方去找；在資源許可情形下，採用探索式的方式去找解) v.s. Exploitation(會往mean小的地方去找；會往機率較高的地方找解)
		
		b-2. covariance function(kernel function:用以描述點與點之間的關係；舉例來說在SVM中會用該函數將低維度映射到高維度空間，以達到線性不可分至線性可分)和mean function會決定某一高斯分布

		b-3. 隨意給定任一超參數(x)，會得loss function(y)，抽一個點會得一條震盪的線 (每一條震盪的線都代表一個樣本，Gaussisan Process)




