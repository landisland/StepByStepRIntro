# Chapter 4 Classification

## Conceptual

1. Using a little bit of algebra, prove that $(4.2)$ is equivalent to $(4.3)$. In other words, the logistic function representation and logit representation for the logistic regression model are equivalent.

$$
\begin{aligned}
&p(X)=\frac{Z}{1+Z} \\
&\frac{1}{p(X)}=\frac{1+Z}{Z}=1+\frac{1}{Z} \\
&Z=\frac{1}{\frac{1}{p(X)}-1}=\frac{1}{\frac{1-p(X)}{p(X)}}=\frac{p(X)}{1-p(X)}
\end{aligned}
$$

2. It was stated in the text that classifying an observation to the class for which (4.17) is largest is equivalent to classifying an observation to the class for which (4.18) is largest. Prove that this is the case. In other words, under the assumption that the observations in the $k$ th class are drawn from a $N\left(\mu_{k}, \sigma^{2}\right)$ distribution, the Bayes classifier assigns an observation to the class for which the discriminant function is maximized.

* Equation becomes $p_k(x) = C \pi_k \exp(- \frac {1} {2 \sigma^2} (\mu_k^2 - 2x \mu_k))$
* Take log of both sides $log(p_k(x)) = log(C) + log(\pi_k) + (- \frac {1} {2 \sigma^2} (\mu_k^2 - 2x \mu_k))$
* Simplify and rearrange $log(p_k(x)) =  (\frac {2x \mu_k} {2 \sigma^2} -\frac {\mu_k^2} {2 \sigma^2}) + log(\pi_k) + log(C)$

3. This problem relates to the QDA model, in which the observations within each class are drawn from a normal distribution with a classspecific mean vector and a class specific covariance matrix. We consider the simple case where $p=1 ;$ i.e. there is only one feature.

   Suppose that we have $K$ classes, and that if an observation belongs to the $k$ th class then $X$ comes from a one-dimensional normal distribution, $X \sim N\left(\mu_{k}, \sigma_{k}^{2}\right)$. Recall that the density function for the one-dimensional normal distribution is given in $(4.16)$. Prove that in this case, the Bayes classifier is not linear. Argue that it is in fact quadratic.

If $\sigma$ varies by $k$ then Equation (4.12) becomes: $p_k(x) = \frac {\pi_k \frac {1} {\sqrt{2 \pi} \sigma_k} \exp(- \frac {1} {2 \sigma_k^2} (x - \mu_k)^2) } {\sum { \pi_l \frac {1} {\sqrt{2 \pi} \sigma_k} \exp(- \frac {1} {2 \sigma_k^2} (x - \mu_l)^2) }}$

The constant term that does not vary by $k$ becomes: $C' = \frac { \frac {1} {\sqrt{2 \pi}}} {\sum { \pi_l \frac {1} {\sqrt{2 \pi} \sigma_k} \exp(- \frac {1} {2 \sigma_k^2} (x - \mu_l)^2) }}$



* Equation becomes $p_k(x) = C' \frac{\pi_k}{\sigma_k} \exp(- \frac {1} {2 \sigma_k^2} (x - \mu_k)^2)$
* Take log of both sides  $log(p_k(x)) = log(C') + log(\pi_k) - log(\sigma_k) + (- \frac {1} {2 \sigma_k^2} (x - \mu_k)^2)$
* Simplify and rearrange $log(p_k(x)) = (- \frac {1} {2 \sigma_k^2} (x^2 + \mu_k^2 - 2x\mu_k)) + log(\pi_k) - log(\sigma_k) + log(C')$

There's the $x^2$.



5. We now examine the differences between LDA and QDA.
  (a) If the Bayes decision boundary is linear, do we expect LDA or QDA to perform better on the training set? On the test set?

  * If the actual decision boundary is linear, then we would expect LDA to perform better on the test set. For the training set, QDA has a chance of performing better if it overfits.

  (b) If the Bayes decision boundary is non-linear, do we expect LDA or QDA to perform better on the training set? On the test set?

  * QDA would likely perform better on both the training set and the test set. 

  (c) In general, as the sample size $n$ increases, do we expect the test prediction accuracy of QDA relative to LDA to improve, decline, or be unchanged? Why?

  * In general a large sample size is more beneficial for QDA so would expect QDA accuracy to increase more than LDA. 

  (d) True or False: Even if the Bayes decision boundary for a given problem is linear, we will probably achieve a superior test error rate using QDA rather than LDA because QDA is flexible enough to model a linear decision boundary. Justify your answer.

  * FALSE: We might achieve a better error rate on the training set but not on the test set because if the true decision boundary is linear then the QDA is not flexible in any predictive way.

6. Suppose we collect data for a group of students in a statistics class with variables $X_{1}=$ hours studied, $X_{2}=$ undergrad GPA, and $Y=$ receive an A. We fit a logistic regression and produce estimated coefficient, $\hat{\beta}_{0}=-6, \hat{\beta}_{1}=0.05, \hat{\beta}_{2}=1$.
  (a) Estimate the probability that a student who studies for $40 \mathrm{~h}$ and has an undergrad GPA of $3.5$ gets an A in the class.

  * For logistic regression, $p(X) = \frac{e^{\beta_0+\beta_1 X_1+\beta_2 X_2}}{1+e^{\beta_0+\beta_1 X_1+\beta_2 X_2}}$
  * Plugging in the values $p(X) = \frac{e^{-6 + 0.05 \times 40 + 1 \times 3.5}}{1+e^{-6+0.05 \times 40 + 1 \times 3.5}} =0.38$

  (b) How many hours would the student in part (a) need to study to have a $50 \%$ chance of getting an A in the class?

  * Solve this equation $0.5 = \frac{e^{-6 + 0.05 X_1 + 1 \times 3.5}}{1+e^{-6+0.05 X_1 + 1 \times 3.5}}$
  * Which equates to solving the logit equation $log(\frac{0.5}{1-0.5}) = -6 + 0.05 X_1 + 1 \times 3.5$ = 50

  

7. Suppose that we wish to predict whether a given stock will issue a dividend this year ("Yes" or "No") based on $X$, last year's percent profit. We examine a large number of companies and discover that the mean value of $X$ for companies that issued a dividend was $\bar{X}=10$ while the mean for those that didn't was $\bar{X}=0$. In addition, the variance of $X$ for these two sets of companies was $\hat{\sigma}^{2}=36$. Finally, $80 \%$ of companies issued dividends. Assuming that $X$ follows a normal distribution, predict the probability that a company will issue a dividend this year given that its percentage profit was $X=4$ last year.
   * For constant variance, $p_k(x) = \frac {\pi_k \frac {1} {\sqrt{2 \pi} \sigma} \exp(- \frac {1} {2 \sigma^2} (x - \mu_k)^2) } {\sum { \pi_l \frac {1} {\sqrt{2 \pi} \sigma} \exp(- \frac {1} {2 \sigma^2} (x - \mu_l)^2) }}$
   * Evaluating this becomes $p_{yes}(4) = \frac {0.8 \exp(- \frac {1} {2 \times 36} (4 - 10)^2)} {0.8 \exp(- \frac {1} {2 \times 36} (4 - 10)^2) + (1-0.8) \exp(- \frac {1} {2 \times 36} (4 - 0)^2)}$= 75.2%



8. Suppose that we take a data set, divide it into equally-sized training and test sets, and then try out two different classification procedures. First we use logistic regression and get an error rate of $20 \%$ on the training data and $30 \%$ on the test data. Next we use 1-nearest neighbors (i.e. $K=1$ ) and get an average error rate (averaged over both test and training data sets) of $18 \%$. Based on these results, which method should we prefer to use for classification of new observations? Why?
   * There's not enough information to say which method is better. With such a high error rate for the logistic regression, it's possible that the true decision boundary is not linear, so KNN=1 might have a better fit. On the other hand, KNN=1 has a high propensity to overfit. With KNN=1 having an average error of 18%, it's possible that the training error is close to 0% and the test error is more than 30%. If we are selecting the model with only error rate data, then we want to know which model has the lower __test__ error rate.

9. This problem has to do with odds.
  (a) On average, what fraction of people with an odds of $0.37$ of defaulting on their credit card payment will in fact default?

  * We want to solve $0.37 = \frac{p_{default}}{1-p_{default}}$
  * Rearranging, this becomes $\frac{1}{0.37} = \frac{1-p_{default}}{p_{default}} = \frac{1}{p_{default}}-1$
  * Finally $p_{default} = \frac{1}{\frac{1}{0.37}+1}$=27%

  (b) Suppose that an individual has a $16 \%$ chance of defaulting on her credit card payment. What are the odds that she will default?

  * $1/(1/0.16+1)=0.19$





