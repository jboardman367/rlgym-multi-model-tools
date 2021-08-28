# rlgym-multi-model-tools
Tools for rlgym-tools for training multiple models at once, and training against other models

Can handle pretty much any arrangement of models into the match, and can mask certain models
out of the learning, as well as having seperate reward functions for each model.

The example is basically just the example from rlgym-tools, but showing the extra features
this module adds.

PRO TIP: Try and minimise the amount of models active at once, as the more active models there
are, the slower the learning will run.
