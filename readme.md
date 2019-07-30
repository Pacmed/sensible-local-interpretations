# class weight uncertainty


*Note: this repo is actively maintained. For any questions please file an issue.*

This project aims to provide a way to interpret individual predictions made by a model in terms of 3 things: (1) uncertainty, (2) contribution, and (3) sensitivity. The methods here are model-agnostic and fast.

![](results/illustrate/illustration.jpg)

The outcome allows for an interactive exploration of how a model makes its prediction:

http://htmlpreview.github.io/?https://github.com/csinva/class-weight-uncertainty/results/interp/out_breast_cancer.html

# uncertainty

- in addition to original model, train model with one class upweighted, one class downweighted
- use these additional models to get info about uncertainty (mostly aleatoric)
- can define uncertainty as overconfident prediction - underconfident prediction

# contribution

- how does this prediction differ from a typical prediction?


# sensitivity
- what is the outcome of changing this feature?

