# class weight uncertainty

## todo

- compare w/ asymmetric entropy
- show basic sims / illustrations on why this would work
- try on more dsets: simulations, mnist / fashion mnist
- try for both nns and things like decision trees.

## basic idea

- in addition to original model, train model with one class upweighted, one class downweighted
- use these additional models to get info about uncertainty (mostly aleatoric)
- can define uncertainty as overconfident prediction - underconfident prediction

## extensions

- look at feature importance difference between overconfident / underconfident
- theoretical grounding?
- connection to quantile regression
