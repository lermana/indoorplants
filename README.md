# modeling

This is a Python library of helper functions and workflow automaters that I use to make my life easier, the first installment of which is **evaluation**.

### evaluation

A module of functions that allow for streamlined and tidy model evaluation, for use with models from *sci-kit-learn*.

This is a first pass; the code works, but has not been optimized. Additionally, some changes could be made here for style and clarity.

Some key functions are:

**five_fold_validation**: allows for automated model evaluation. 

**validate_multiple_models**: runs *five_fold_model_validation* for multiple models, returning aggregate results. 

More information on these functions can be found on their respective pages on the wiki. 

Additional functions include decomposed routines that are used by the aforementioned validation functions, as well as some wrappers for sk-learn's *confusion_matrix* that can be used for evaluating the performance of classifiers.
    
This code was written using *Python 3.5*, using *Pandas 0.19.2*, *sci-kit learn 0.18.1*, and *NumPy 1.12.0*, I can't think of anything here that requires Pandas .19, and ditto for NumPy 1.12. If using an older version of sci-kit learn however, the confusion_matrix function will have to be imported from *metrics* and not *model_selection*.	