== Code for paper titled ``On the Long-term Impact of Algorithmic Policies: Effort Unfairness and Feature Segregation through the Lens of Social Learning`` to appear at ICML 2019 ==

Authors:
Vedant Nanda, MPI-SWS
Hoda Heidari, ETH Zürich
Krishna P. Gummadi, MPI-SWS


Requirements:

python 3.5.3 or above

Python Libraries needed to run the code:

asn1crypto==0.24.0
bcrypt==3.1.5
cffi==1.11.5
cryptography==2.4.2
cycler==0.10.0
idna==2.8
kiwisolver==1.0.1
matplotlib==3.0.2
numpy==1.16.0
pandas==0.23.4
paramiko==2.4.2
pkg-resources==0.0.0
pyasn1==0.4.5
pycparser==2.19
pydotplus==2.0.2
PyNaCl==1.3.0
pyparsing==2.3.0
python-dateutil==2.7.5
pytz==2018.9
scikit-learn==0.20.2
scipy==1.2.0
seaborn==0.9.0
six==1.12.0
sklearn==0.0
xlrd==1.2.0


To install all these, cd to effort_reward_fairness and execute ``pip install -r requirements.txt``. You might need sudo permission for installation.


= Reproduction of experiments and results in the paper titled ``On the Long-term Impact of Algorithmic Policies: Effort Unfairness and Feature Segregation through the Lens of Social Learning`` =

Once dependencies have been installed, cd to the ``effort_reward_fairness`` directory.

1. We first need to generate explanations for users in both the train and the test set. To do so, run ``python experiment.py train`` and ``python experiment.py test``. These can be run either simultaneously or sequentially (since they generate explanations for different parts of the dataset). However, you must run both these commands before proceeding to the next step. As a consequence of running ``python experiment.py test`` the plot containing analysis of different disparity measure for different models is generated and stored in ``./effort_reward_fairness/results/StudentPerf/disparity_plots/all_disp_in_one_test.pdf``. This is the plot in Fig 3 in the paper.

2. To generate plot in Fig 1 and 2, run ``python effort_reward_function_plots.py train``. This will generate Average Reward as a function of effort and Average effort as a function of reward. You will find ``Effort_vs_Average_Reward_together.pdf`` and ``Reward_vs_Average_Effort_together.pdf`` in the directory ``./effort_reward_fairness/results/StudentPerf/disparity_plots/``. These can also be seen in Fig 1 and 2 in the paper.

3. Once you have completed step 1, then run ``python long_term_impact.py``. This will generate new set of feature vectors which correspond to action taken by users on explanations given in step 1.

4. After step 3 completes, execute ``python utility_thresholds.py`` to see the effect of algorithmic policies on segregation. As a result of executing ``python utility_thresholds.py`` you will find ``segregation_centralization.pdf``, ``segregation_atkinson.pdf``, ``segregation_ACI.pdf`` in the directory ``./effort_reward_fairness/results/StudentPerf/segregation_plots``. These are included in Fig 4 in the paper.

5. Now to generate plots for models with Fairness Constraints, open experiment.py and change the global variable ``FAIRNESS_CONSTRAINTS`` to True. Save the file. Now repeat steps 1, 3 and 4 listed above (step 2 is optional, results of step 2 for models with Fairness Constraints are not included in the paper). **

6. Once you have done step 5, you'll find ``segregation_centralization_fc.pdf``, ``segregation_atkinson_fc.pdf``, ``segregation_ACI_fc.pdf`` in the directory ``./effort_reward_fairness/results/StudentPerf/segregation_plots``. Thes plots are included in Fig 5 in the paper.


== Preprocessing ==

Code for preprocessing the Student Performance Dataset (http://archive.ics.uci.edu/ml/datasets/Student+Performance) can be found in ``./util/datasets/data/student_performance/preprocessing.ipynb``. An executable python file (``preprocessing.py``) of this notebook is also available in the same directory.


** To run Fairness Constraints you need the file ``trained_linregfc_StudentPerf.mat`` in the ``./effort_reward_fairness`` directory. For easy reproduction of results we have included this file, however this is generated using MATLAB code in the directory ``./Fairness_constraints`` and can be found in the directory ``./Fairness_constraints/Output`` after you run ``./Fairness_constraints/Social_welfare_constrained_ERM_regularized.m``.
