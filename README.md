IdentifyLiarsHiders
====================

This is an **Octave project** where we use **Machine Learning** in order to identify a particular type of agents.

Sometimes, during a multi-agent argumentation debate, some agents **lie or hide** important arguments, because they have personal reasons to do so.
We propose a way to analyze the trace of an argumentation debate, in order to identify liar and hiding agents.
The Machine Learning algorithm used here is **Logistic Regression**. It learns from a training set, and it is evaluated with respect to a test set.
Both sets are summarizations of a large number of debate simulations.
Such debate simulations can be made, for example, by using our code in the *ArgumentationDebates* repository.

The main idea of this work is the following:
From all the argument, attack and support insertions made by the agents during a debate, we compute (for each agent),
some numerical values for a number of **agent attributes**. We have defined attributes such as **activity**, 
**opinionatedness** and **classifiability** which describe an agent's overall behavior during a debate.

Then, we use Logistic Regression to predict if an agent is a liar or a hider (or both, or neither), based on the numerical values of these attributes.

Finally, we evaluate our predictions, using metrics such as:
* **Precision**
* **Recall**
* **Accuracy**
* **F1-score**
