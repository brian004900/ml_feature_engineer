# PythonProject

# Setup for developement:

- Setup a python 3.x venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
- `pip3 install --upgrade pip`
- Install pip-tools `pip3 install pip-tools`
- Update dev requirements: `pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`
- Update requirements: `pip-compile --output-file=requirements.txt requirements.in --upgrade`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- Install requirements `pip3 install -r requirements.txt`
- `pre-commit install`

## Update versions

`pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`
`pip-compile --output-file=requirements.txt requirements.in --upgrade`

# Run `pre-commit` locally.

`pre-commit run --all-files`

# Project Name
> **Baseball home team win prediction**

> The project is to predict whether a baseball home team wins. 
Continuing from the previous assignment, the purpose of this final assignment is to predict whether the baseball home team will win or not. So the response will be the home team win(1) or lose(2). I use the [**baseball database**](https://teaching.mrsharky.com/data/baseball.sql.tar.gz) to make multiple predictors which will be represented in the following paragraph. This prediction is an interesting experience. Through this project, I have improved both SQL and python skill. But the output is still far from baseball game gambling.

## Table of Contents
* [Tables](#Tables)
* [Features](#Features)
* [Correlation](#Correlation)
* [Brute Force](#Brute-Force)
* [Random Forest Ranking and p-value](#Random-Forest-Ranking-and-p-value)
* [Train Test split](#Train-Test-split)
* [Models](#Models)
<!-- * [License](#license) -->

## Skills
Python (scikit-learn, pyspark, plotly, statsmodels, pandas, numpy), Docker, Bash, SQL (MariaDB)

## Tables
I use MariaDB, and the SQL script for generating analysis is written in sql.sh bash script. The baseball database tables I use include batter_counts, pitcher_counts, team_results, inning action, team streak, and boxscore. First, we use the team_result table as the main table to be joined. The special thing here is that the team information of the two teams with the same game_id in the team_result must be merged into one row. The team_result table records the victory or defeat of each team in each game.
``` 
select a.game_id as gameid,
       a.local_date as date,
       a.team_id as h_team,
       b.team_id as a_team,
       CASE WHEN a.home_away = 'H' and a.win_lose = 'W' THEN 1
            WHEN a.home_away = 'A' and a.win_lose = 'W' THEN 1
            ELSE 0 END as hometeam_win

from win_lose a
join win_lose b
on a.game_id = b.game_id
where a.team_id != b.team_id and a.home_away = 'H'
order by a.game_id);
``` 
Batter_counts records the data of each batter, here we sum up the batter data in units of each game. In the pitcher_counts table, we only count starting pitchers. The rest of the tables are used to build additional predictors.

The final analysis is the result table. It takes about ten minutes to generate this table using my SQL script. During the join table process, each table has missing game data, so there are 7284 records left in the final result table. This table contains all the original features, and each data contains the predictors of the home team and the away team. 
In terms of data pre-processing, I filled in the average value for all non-value. In addition, since my predictors are all continuous, I set the data type to float to facilitate subsequent analysis and calculation.
<!-- Note -->

## Features
From the original result table, I created a total of [**59** features (page: baseball features)](https://github.com/brian004900/bda602-brian/wiki/Baseball-Features). All features are listed in the above link.

Most of the features are created based on Wikipedia's [baseball statistics](https://en.wikipedia.org/wiki/Baseball_statistics). Among them, there are 48 batting statistics. There are 4 initial pitcher statistics.

In terms of the added feature, I think that the number of pitcher substitutions will affect the outcome of the game, so I built a pitcher substitution predictor. In most cases, if the number of pitcher substitutions is high, it is likely to represent the poor performance of the team's pitchers and thus affect the team's performance. The event column in the inning action records each "pitcher substitution". 

<img src="https://github.com/brian004900/bda602-brian/blob/final/image/pitch%20substitute.png" width="700"/>

Although the performance of this predictor is not good, whether it is in p-value or random forest importance, we can find from the predictor/response plot that indeed as the number of pitcher substitutions increases, the team loses more games than wins There are many sessions.

Then there is the team streak predictor. In order to avoid this predictor or even the team_streak table being cheating, I later used the team_results table which is the table I chose to join all other tables to calculate streak. I recalculated the winning streak and losing streak (losing Streak is sorted as a negative value), using this predictor, the accuracy rate of all models is as high as 90%, the code is as follows.
``` 
create table ws(
SELECT
    T1.game_id as game_id,
    T1.team_id as team_id,
    T1.win_lose as win_lose,
    (
        SELECT count(*)
        FROM team_results T2
        WHERE
            T1.win_lose = T2.win_lose
            AND T1.team_id = T2.team_id
            AND T1.local_date > T2.local_date
            AND NOT EXISTS (
                SELECT *
                FROM team_results T3
                WHERE
                    T2.win_lose <> T3.win_lose
                    AND T1.team_id = T3.team_id
                    AND T3.local_date BETWEEN T2.local_date AND T1.local_date
            )
    ) as counting
FROM team_results T1);
``` 
But I found that even if the victory or loss of the day is deducted, using the above SQL script to calculate the winning streak will still make the table maintain a pattern so that the predictor can know the future, I can say it is cheating. From the table below, we can know how many consecutive victories before playing the game. When the team ends losing or winning, the winning or losing streak ends and recounts! This is what causes cheating.

<img src="https://github.com/brian004900/bda602-brian/blob/final/image/streak_cheat.png" width="700"/>

I added the feature of temperature after the presentation. I think that overheating temperature will affect the team's performance. I first used the temperature of the day because I think it is very accurate in terms of the current weather forecast. Using this predictor, the random forest The accuracy rate increased by 2% to 3%. However, this is still cheating so I finally decided to calculate the average temperature of the previous 10 days, and this predictor still achieves good results.

<img src="https://github.com/brian004900/bda602-brian/blob/final/image/temprf.png" width="700"/>

<img src="https://github.com/brian004900/bda602-brian/blob/final/image/tempmean.png" width="700"/>

## Correlation
<img src="https://github.com/brian004900/bda602-brian/blob/final/image/corr.png" width="700"/>

Removing highly correlated features can reduce noise and increase the quality of the model. I have removed many features so that the correlation between the two features will not be higher than 90%. Before removing the feature, I will first observe brute force to determine whether the brute force plot has a particulate pattern to experiment with new features. Then, I will decide which features to remove based on p-value and random forest importance.

## Brute Force
When two features have a certain correlation, I will observe the brute force plot. Take the home team batting average and away team batting average as examples, from their brute force pattern we can observe that when one value is high, the other value is low. At this time, I will subtract these two features.

<img src="https://github.com/brian004900/bda602-brian/blob/final/image/battingavg.png" width="700"/>

I used this method to create 20 more new features, and among the 20, 17 achieved good results. The feature names I created start with diff, from the next [session](#Random-Forest-Ranking-and-p- value), it can be found that these features perform well in random forest importance. In the original data without removing any features, their values are higher than the average, and even the p-value of several features is also lower than 0.05. Although the final performance of these features is not obvious, but the accuracy of all models There is indeed a slight improvement, especially logistic regression.

## Random Forest Ranking and p-value
Finally, I remove more features based on random forest importance and p-value. The principle of removal is to satisfy both when the p-value is higher than 0.05 and the value of the random forest is relatively unimportant. The following charts are the random forest importance plot and p-value of all the features I finally decided to use.

<img src="https://github.com/brian004900/bda602-brian/blob/final/image/rf.png" width="700"/>

<img src="https://github.com/brian004900/bda602-brian/blob/final/image/ftable.png" width="700"/>

## Rolling Average
In the rolling average part, according to the original 100 days of work, the accuracy of my random forest is less than 50%. When adjusted for 10 days, the average accuracy increased by 3% to 5%. The reason comes from the recent team performance, including the team list, which can better reflect the current state of the team, so the prediction will be more accurate. Additionally, I looked at the data and found that the same two teams have a high probability of back-to-back games on different days, which leads to overwhelming teams winning back-to-back. The 10-day rolling average can better reflect the relationship between teams than the 100-day rolling average, so it can improve the prediction accuracy.

## Train Test split
All predictions are sequential, and future predictions must be based on past historical records, so I did not disrupt the order during the train test split and set the ratio of the train test to 77:33.

## Models

<img src="https://github.com/brian004900/bda602-brian/blob/final/image/model.png" width="700"/>

To sum up, after feature engineering and adjustment, the accuracy of logistic regression has the most obvious increase, the value is nearly 10%, and the final accuracy is also close to 55%. The accuracy of random forest cannot perform better because no decisive feature has been created. According to [F-score](https://en.wikipedia.org/wiki/F-score), logistic regression is indeed more stable.


