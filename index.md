## Machine Learning Project: Identify Women’s Risks of Intimate Partner Violence with Evidence from South East Asia


### Background and Solution Overview

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for
- Intimate partner violence (IPV) is one of the most common forms of violence against women and includes physical, sexual, and emotional abuse.
- South East Asia is one of the areas with highest prevalence of physical and sexual IPV of around 37%.
- The goal of this study are:
  - To report the prevalence of IPV among ever married or cohabiting women in Pakistan, Cambodia, Philippines, Maldives and Nepal
  - To develop classification algorithms on individual-level variables and couple-level variables to predict whether a specific woman is prone to IPV in these five countries;
  - To identify important features associated with experiencing physical, sexual or emotional IPV.


### Data Source

- Main data source is Demographic and Health Surveys (DHS) conducted by the United States Agency of International Development
- DHS program had developed a standard module and methodology for the collection of data on domestic violence by 2000.
- The standard module was used in all of the countries examined in this report: Cambodia(2014), Maldives(2016), Nepal(2016), Pakistan(2017), Philippines(2017)
- We filtered data using the variable if_union to only include women who are currently or formerly married (or live with a partner)
 
 
 ### Machine Learning Approach
- We developed binary classification - algorithms with individual-level variables and couple-level variables to predict whether a specific ever married or cohabiting woman is prone to each category of IPV in Pakistan, Cambodia, Philippines, Maldives and Nepal.
- Using ten-fold cross-validation, we trained Balanced Random Forest, Weighted Random Forest, Decision Tree, Random Forest, Logistic Regression, Linear Support Vector Machines and Gaussian Naive Bayes models with Synthetic Minority Over-sampling Technique


