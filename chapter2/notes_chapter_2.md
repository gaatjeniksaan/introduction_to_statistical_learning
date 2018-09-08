### CHAPTER 2

#### Regression

The goal with statistical models is to do a prediction or to infer the underlying relationship. 

Parametric methods assume a functional form of f(x), e.g. a linear relationship.

Non-parametric methods do not assume such a thing, and thus have more freedom to fit the data. Generally more data is needed in these cases. 

Always ask yourself what is important: interpretability or flexibility when choosing a model. 

MSE = 1/n SUM (yi - fhat(xi))^2

The MSE can be decomposed (for a single test point in this case) into:
E(y0 - fhat(x0))^2 = VAR(fhat(x0)) + [BIAS(fhat(x0))]^2 + VAR(error)

VAR(error) is always positive and called the reducible error. VAR + BIAS^2 term is the reducible error.

Variance interpretation: indicative of how much fhat would change if we used a different data training set.
Bias interpretation: indicative of the error introduced by underlying assumptions and model choice. E.g. assuming linearity when the problem is in fact not linear.

| Flexibility | Variance | Bias |
|-------------|----------|------|
| High        | High     | Low  |
| Low         | Low      | High |

#### Classification
Accuracy of classification:
1/n SUM I(yi != yhati)

Bayes classifier assigns class for which:
Pr(Y=j | X=x0) is largest

This assumes the probability distribution of the classes is known. 
The corresponding error rate is given by:

1 - maxj Pr(Y=j | X=x0)
Or in words, the probability of miss classifying. 

Ideally we'd always use a Bayes classifier, but this is not feasible because the probabilities of the classes are not known. This makes the Bayes classifier the unattainable gold standard.

K Nearest Neighbors (KNN) implements *estimated* probability. For test point x0, K clusters and N0 closest points:

Pr(Y=j | X=x0) = 1/K SUM I(yi = j)

Low K give the model a lot of flexibility but high variance (overfitting). High K make the model more rigid (high bias).
