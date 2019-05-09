# machine_learning
Repository with my machine learning projects

In this repository I available my machine learning projects and codes.

- TraderBot
In this project I developed a machine learning model that predicts the future price of a stock into financial market.
For this project I used Random Forest model to predicts the future price.
Steps:
1- EA_Random_forest.mq5(MetaTrader5 file) send stock quotes to server.py
2- server.py receive the market data, train, predicts the future price and send a signal to MetaTrader
3- EA_Random_forest.mq5 receive the signal and send the order to the market.

- Blackjack_agent.py
In this project I developed a intelligent agent that learn how to play blackjack. 
For this project I used reinforced learning techniques to teach the agent how to play. 
The environment used was available by OpenAi gym.


- Frozen_lake.py
In this project I developed a intelligent agent that learn how to walk on a frozen lake without falling on the bunk. 
For this project I used reinforced learning techniques to teach the agent how to play. 
The environment used was available by OpenAi gym.


- Santander Customer Satisfaction
Santander.ipynb
Competition from Kaggle to identify the level of customer's satisfaction of Bank Santander.
