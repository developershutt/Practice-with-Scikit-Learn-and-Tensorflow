# What is Machine Learning?
Machine Learning is the science (and art) of programming computers so they can learn from data. 

Here is a slightly more general definition:

Machine Learning is the] field of study that gives computers the ability to learn without being explicitly programmed.

And a more engineering-oriented one: 

A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

For example, your spam filter is a Machine Learning program that can learn to flag spam given examples of spam emails (e.g., flagged by users) and examples of regular (nonspam, also called “ham”) emails. The examples that the system uses to learn are called the training set. Each training example is called a training instance (or sample). In this case, the task T is to flag spam for new emails, the experience E is the training data, and the performance measure P needs to be defined; for example, you can use the ratio of correctly classified emails. This particular performance measure is called accuracy and it is often used in classification tasks. 

# Why Use Machine Learning? 
Consider how you would write a spam filter using traditional programming techniques:
   1. First you would look at what spam typically looks like. You might notice that some words or phrases (such as “4U,” “credit card,”   “free,” and “amazing”) tend to come up a lot in the subject. Perhaps you would also notice a few other patterns in the sender’s name, the email’s body, and so on. 
   2. You would write a detection algorithm for each of the patterns that you noticed, and your program would flag emails as spam if a number of these patterns are detected. 
   3. You would test your program, and repeat steps 1 and 2 until it is good enough.
![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img1.PNG)

Since the problem is not trivial, your program will likely become a long list of complex rules — pretty hard to maintain.
In contrast, a spam filter based on Machine Learning techniques automatically learns which words and phrases are good predictors of spam by detecting unusually frequent patterns of words in the spam examples compared to the ham examples.The program is much shorter, easier to maintain, and most likely more accurate.

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img2.PNG)

Moreover, if spammers notice that all their emails containing “4U” are blocked, they might start writing “For U” instead. A spam filter using traditional programming techniques would need to be updated to flag “For U” emails. If spammers keep working around your spam filter, you will need to keep writing new rules forever.
In contrast, a spam filter based on Machine Learning techniques automatically notices that “For U” has become unusually frequent in spam flagged by users, and it starts flagging them without your intervention.

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img3.PNG)

Another area where Machine Learning shines is for problems that either are too complex for traditional approaches or have no known algorithm. For example, consider speech recognition: say you want to start simple and write a program capable of distinguishing the words “one” and “two.” You might notice that the word “two” starts with a high-pitch sound (“T”), so you could hardcode an algorithm that measures high-pitch sound intensity and use that to distinguish ones and twos. Obviously this technique will not scale to thousands of words spoken by millions of very different people in noisy environments and in dozens of languages. The best solution (at least today) is to write an algorithm that learns by itself, given many example recordings for each word.


Finally, Machine Learning can help humans learn ML algorithms can be inspected to see what they have learned (although for some algorithms this can be tricky). For instance, once the spam filter has been trained on enough spam, it can easily be inspected to reveal the list of words and combinations of words that it believes are the best predictors of spam. Sometimes this will reveal unsuspected correlations or new trends, and thereby lead to a better understanding of the problem. 
Applying ML techniques to dig into large amounts of data can help discover patterns that were not immediately apparent. This is called data mining.

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img4.PNG)

To summarize, Machine Learning is great for:
  * Problems for which existing solutions require a lot of hand-tuning or long lists of rules: one Machine Learning algorithm can often simplify code and perform better. 
  * Complex problems for which there is no good solution at all using a traditional approach: the best Machine Learning techniques can find a solution. 
  * Fluctuating environments: a Machine Learning system can adapt to new data. 
  * Getting insights about complex problems and large amounts of data.
  
  
# Types of Machine Learning Systems
There are so many different types of Machine Learning systems that it is useful to classify them in broad categories based on:

  * Whether or not they are trained with human supervision (supervised, unsupervised, semisupervised, and Reinforcement Learning)
  * Whether or not they can learn incrementally on the fly (online versus batch learning) .
  * Whether they work by simply comparing new data points to known data points, or instead detect patterns in the training data and build a predictive model, much like scientists do (instance-based versus model-based learning) 
  
These criteria are not exclusive; you can combine them in any way you like. For example, a state-of-the-art spam filter may learn on the fly using a deep neural network model trained using examples of spam and ham; this makes it an online, modelbased, supervised learning system. 

Let’s look at each of these criteria a bit more closely.

# Supervised/Unsupervised Learning
Machine Learning systems can be classified according to the amount and type of supervision they get during training. There are four major categories: supervised learning, unsupervised learning, semisupervised learning, and Reinforcement Learning. 

  # Supervised Learning
  In supervised learning, the training data you feed to the algorithm includes the desired solutions, called labels 
  
  ![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img5.PNG)
  A labeled training set for supervised learning (e.g., spam classification)
  
A typical supervised learning task is classification. The spam filter is a good example of this: it is trained with many example emails along with their class (spam or ham), and it must learn how to classify new emails. 
  
Another typical task is to predict a target numeric value, such as the price of a car, given a set of features (mileage, age, brand, etc.) called predictors. This sort of task is called regression.To train the system, you need to give it many examples of cars, including both their predictors and their labels (i.e., their prices).

*Note: In Machine Learning an attribute is a data type (e.g., “Mileage”), while a feature has several meanings depending on the context, but generally means an attribute plus its value (e.g., “Mileage = 15,000”). Many people use the words attribute and feature interchangeably, though.

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img6.PNG)
  Regression
Note that some regression algorithms can be used for classification as well, and vice versa. For example, Logistic Regression is commonly used for classification, as it can output a value that corresponds to the probability of belonging to a given class (e.g., 20% chance of being spam). 

Here are some of the most important supervised learning algorithms:
  * k-Nearest Neighbores
  * Linear Regression
  * Logistic Regression
  * Support Vector Machine (SVMs)
  * Decision Tree and Random Forest
  * Neural Networks
  
  # Unsupervised learning
  In unsupervised learning, as you might guess, the training data is unlabeled. The system tries to learn without a teacher.
  
  ![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img7.PNG)
  
  Here are some of the most important unsupervised learning algorithms:
    * Clustering
      * k-Means
      * Hierarchical Cluster Analysis (HCA)
      * Expectation Maximization
      
   * Visualization and dimensionality reduction 
      
      * Principal Component Analysis (PCA)
      * Kernel PCA
      * Locally-Linear Embedding (LLE)
      * t-distributed Stochastic Neighbor Embedding (t-SNE)
      
   * Association rule learning
   
      * Apriori
      * Eclat
      
For example, say you have a lot of data about your blog’s visitors. You may want to run a clustering algorithm to try to detect groups of similar visitors. At no point do you tell the algorithm which group a visitor belongs to: it finds those connections without your help. For example, it might notice that 40% of your visitors are males who love comic books and generally read your blog in the evening, while 20% are young sci-fi lovers who visit during the weekends, and so on. If you use a hierarchical clustering algorithm, it may also subdivide each group into smaller groups. This may help you target your posts for each group.
 
![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img8.PNG)

Visualization algorithms are also good examples of unsupervised learning algorithms: you feed them a lot of complex and unlabeled data, and they output a 2D or 3D representation of your data that can easily be plotted. These algorithms try to preserve as much structure as they can (e.g., trying to keep separate clusters in the input space from overlapping in the visualization), so you can understand how the data is organized and perhaps identify unsuspected patterns.

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img9.PNG)

Example of a t-SNE visualization highlighting semantic clusters.

A related task is dimensionality reduction, in which the goal is to simplify the data without losing too much information. One way to do this is to merge several correlated features into one. For example, a car’s mileage may be very correlated with its age, so the dimensionality reduction algorithm will merge them into one feature that represents the car’s wear and tear. This is called feature extraction.

  *Tip: It is often a good idea to try to reduce the dimension of your training data using a dimensionality reduction algorithm before you feed it to another Machine Learning algorithm (such as a supervised learning algorithm). It will run much faster, the data will take up less disk and memory space, and in some cases it may also perform better.

Yet another important unsupervised task is anomaly detection — for example, detecting unusual credit card transactions to prevent fraud, catching manufacturing defects, or automatically removing outliers from a dataset before feeding it to another learning algorithm. The system is trained with normal instances, and when it sees a new instance it can tell whether it looks like a normal one or whether it is likely an anomaly.

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img10.PNG)


Finally, another common unsupervised task is association rule learning, in which the goal is to dig into large amounts of data and discover interesting relations between attributes. For example, suppose you own a supermarket. Running an association rule on your sales logs may reveal that people who purchase barbecue sauce and potato chips also tend to buy steak. Thus, you may want to place these items close to each other. 

# Semi-Supervised Learning
Some algorithms can deal with partially labeled training data, usually a lot of unlabeled data and a little bit of labeled data. This is called semisupervised learning.


Some photo-hosting services, such as Google Photos, are good examples of this. Once you upload all your family photos to the service, it automatically recognizes that the same person A shows up in photos 1, 5, and 11, while another person B shows up in photos 2, 5, and 7. This is the unsupervised part of the algorithm (clustering). Now all the system needs is for you to tell it who these people are. Just one label per person,4 and it is able to name everyone in every photo, which is useful for searching photos.

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img11.PNG)


# Reinforcement Learning
Reinforcement Learning is a very different beast. The learning system, called an agent in this context, can observe the environment, select and perform actions, and get rewards in return (or penalties in the form of negative rewards). It must then learn by itself what is the best strategy, called a policy, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img12.PNG)

For example, many robots implement Reinforcement Learning algorithms to learn how to walk. DeepMind’s AlphaGo program is also a good example of Reinforcement Learning: it made the headlines in March 2016 when it beat the world champion Lee Sedol at the game of Go. It learned its winning policy by analyzing millions of games, and then playing many games against itself. Note that learning was turned off during the games against the champion; AlphaGo was just applying the policy it had learned.

# Batch and Online Learning
Another criterion used to classify Machine Learning systems is whether or not the system can learn incrementally from a stream of incoming data. 

# Batch learning

In batch learning, the system is incapable of learning incrementally: it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called offline learning.

If you want a batch learning system to know about new data (such as a new type of spam), you need to train a new version of the system from scratch on the full dataset (not just the new data, but also the old data), then stop the old system and replace it with the new one.
Fortunately, the whole process of training, evaluating, and launching a Machine Learning system can be automated fairly easily, so even a batch learning system can adapt to change. Simply update the data and train a new version of the system from scratch as often as needed. 
This solution is simple and often works fine, but training using the full set of data can take many hours, so you would typically train a new system only every 24 hours or even just weekly. If your system needs to adapt to rapidly changing data (e.g., to predict stock prices), then you need a more reactive solution. 
Also, training on the full set of data requires a lot of computing resources (CPU, memory space, disk space, disk I/O, network I/O, etc.). If you have a lot of data and you automate your system to train from scratch every day, it will end up costing you a lot of money. If the amount of data is huge, it may even be impossible to use a batch learning algorithm. 

Finally, if your system needs to be able to learn autonomously and it has limited resources (e.g., a smartphone application or a rover on Mars), then carrying around large amounts of training data and taking up a lot of resources to train for hours every day is a showstopper.

Fortunately, a better option in all these cases is to use algorithms that are capable of learning incrementally.

# Online Learning
In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or by small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives 

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img13.PNG)
  *Online Learning
  
Online learning is great for systems that receive data as a continuous flow (e.g., stock prices) and need to adapt to change rapidly or autonomously. It is also a good option if you have limited computing resources: once an online learning system has learned about new data instances, it does not need them anymore, so you can discard them (unless you want to be able to roll back to a previous state and “replay” the data). This can save a huge amount of space. 
Online learning algorithms can also be used to train systems on huge datasets that cannot fit in one machine’s main memory (this is called out-of-core learning). The algorithm loads part of the data, runs a training step on that data, and repeats the process until it has run on all of the data.

# Warning
This whole process is usually done offline (i.e., not on the live system), so online learning can be a confusing name. Think of it as incremental learning.

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img4.PNG)

One important parameter of online learning systems is how fast they should adapt to changing data: this is called the learning rate. If you set a high learning rate, then your system will rapidly adapt to new data, but it will also tend to quickly forget the old data (you don’t want a spam filter to flag only the latest kinds of spam it was shown). Conversely, if you set a low learning rate, the system will have more inertia; that is, it will learn more slowly, but it will also be less sensitive to noise in the new data or to sequences of nonrepresentative data points. 
A big challenge with online learning is that if bad data is fed to the system, the system’s performance will gradually decline. If we are talking about a live system, your clients will notice. For example, bad data could come from a malfunctioning sensor on a robot, or from someone spamming a search engine to try to rank high in search results. To reduce this risk, you need to monitor your system closely and promptly switch learning off (and possibly revert to a previously working state) if you detect a drop in performance. You may also want to monitor the input data and react to abnormal data (e.g., using an anomaly detection algorithm).

# Instance Based Vs Model-Based Learning
One more way to categorize Machine Learning systems is by how they generalize. Most Machine Learning tasks are about making predictions. This means that given a number of training examples, the system needs to be able to generalize to examples it has never seen before. Having a good performance measure on the training data is good, but insufficient; the true goal is to perform well on new instances.

There are two main approaches to generalization: instance-based learning and model-based learning.

# Instance -based learning
Possibly the most trivial form of learning is simply to learn by heart. If you were to create a spam filter this way, it would just flag all emails that are identical to emails that have already been flagged by users — not the worst solution, but certainly not the best. 

Instead of just flagging emails that are identical to known spam emails, your spam filter could be programmed to also flag emails that are very similar to known spam emails. This requires a measure of similarity between two emails. A (very basic) similarity measure between two emails could be to count the number of words they have in common. The system would flag an email as spam if it has many words in common with a known spam email. 
This is called instance-based learning: the system learns the examples by heart, then generalizes to new cases using a similarity measure.

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img15.PNG)

# Model-based learning
Another way to generalize from a set of examples is to build a model of these examples, then use that model to make predictions. This is called model-based learning.

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img16.PNG)

For example, suppose you want to know if money makes people happy, so you download the Better Life Index data from the OECD’s website as well as stats about GDP per capita from the IMF’s website. Then you join the tables and sort by GDP per capita. Below tabel shows and excerpt of what you get.

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img17.PNG)

Let’s plot the data for a few random countries

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img18.PNG)

There does seem to be a trend here! Although the data is noisy (i.e., partly random), it looks like life satisfaction goes up more or less linearly as the country’s GDP per capita increases. So you decide to model life satisfaction as a linear function of GDP per capita. This step is called model selection: you selected a linear model of life satisfaction with just one attribute, GDP per capita 

A simple linear model
![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img19.PNG)
This model has two model parameters, θ0 and θ1. By tweaking these parameters, you can make your model represent any linear function

![alt text](https://github.com/manish29071998/Practice-with-Scikit-Learn-and-Tensorflow/blob/master/images/img20.PNG)
