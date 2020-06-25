
# Cost-sensitive deep forest for price prediction

This is the official clone for the implementation of [Cost-sensitive deep forest for price prediction](https://github.com/machao199271/machao199271.github.io/raw/master/Cost-sensitive%20deep%20forest%20for%20price%20prediction.pdf)[1]. The implementation is flexible enough for modifying the model or fit your own datasets.

Reference: [1] Chao Ma, Zhenbing Liu, Zhiguang Cao, Wen Song, Jie Zhang, Weiliang Zeng. Cost-sensitive Deep Forest for Price Prediction.[J] .Pattern Recognition.

## Files description

'data.csv' is cleaned data of P2P car sharing, note that, not all features are used in the demo. And other datasets can be found in Kaggle.

'costsensitive' is a necessary package for [cost-sentive learning](https://costsensitive.readthedocs.io/en/latest/#).

'run_on_server.py' contains the modified K-means we proposed and some other functions which will be used in the following step.

'cost_sensitive_deep_forest.py' is the method we proposed and traditional deep forest.

'random_forest.py' and 'rotation_forest.py' are the traditional methods. Note that, the code of SVM and MLP is not offered because of their bad performance.
