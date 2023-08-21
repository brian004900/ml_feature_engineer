#!/bin/sh
sleep 10
DATABASE_USER=root
DATABASE_PWD=ma
DATABASE_NAME=baseball

DATABASE_TO_COPY_INTO="baseball"
DATABASE_FILE="baseball.sql"

mysql -u$DATABASE_USER -p$DATABASE_PWD -h mydb -e "show databases;"

if mysql -u$DATABASE_USER -p$DATABASE_PWD -h mydb "use ${DATABASE_NAME}"
then
  echo "${DATABASE_NAME} exists"
else
  echo "${DATABASE_NAME} does not exist (creating it)"
  mysql -u$DATABASE_USER -p$DATABASE_PWD -h mydb -e "CREATE DATABASE ${DATABASE_TO_COPY_INTO};"
  mysql -u$DATABASE_USER -p$DATABASE_PWD -h mydb ${DATABASE_TO_COPY_INTO} < ${DATABASE_FILE}
fi

echo "database created"
mysql -u$DATABASE_USER -p$DATABASE_PWD -h mydb -e "
use baseball;

DROP TABLE IF EXISTS s1;
create table s1(
select
    game_id,
    team_id,
    CASE WHEN win_lose = 'W' THEN SUM(streak)-1
     WHEN win_lose = 'L' THEN SUM(streak)+1 end as streak
from team_streak
group by team_streak_id);

DROP TABLE IF EXISTS w1;
create table w1(
select
game_id,
temp
from boxscore);


/* set 0and1 (lose win) */
DROP TABLE IF EXISTS win_lose;
create table win_lose(
SELECT game_id,
       team_id,
       local_date,
       home_away,
       win_lose
FROM team_results
order by game_id, team_id);

/* h_team a_team to a single row */
DROP TABLE IF EXISTS t0;
create table t0(
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

/* calcultate pitching Substitution */
DROP TABLE IF EXISTS ps1;
create table ps1(
select game_id, half, count(event) as pitcher_change
from inning_action
where event like 'Pitching Substitution'
group by game_id, half);

DROP TABLE IF EXISTS ps2;
create table ps2(
select
  game_id,
  max(case when half = 0 then pitcher_change end) as home_pitch_substute,
  max(case when half = 1 then pitcher_change end) away_pitch_substute
from ps1
group by game_id);

DROP TABLE IF EXISTS t1;
create table t1(
SELECT game_id,
       team_id,
    SUM(Hit) as H,
    SUM(atBat) as AB,
    SUM(Walk) as BB,
    SUM(Home_Run) as HR,
    SUM(Strikeout) as K,
    SUM(Intent_Walk) as IW,
    SUM(plateApperance) as plate_appear,
    SUM(Single) as 1B,
    SUM(batter_counts.Double) as 2B,
    SUM(Triple) as 3B,
    SUM(Sac_Fly) as sf,
    SUM(Ground_Out) as g,
    SUM(Fly_Out) as f,
    SUM(Hit_By_Pitch) as hbp,
    SUM(Sac_Bunt) as sh
FROM batter_counts
GROUP BY game_id, team_id);

DROP TABLE IF EXISTS p1;
create table p1(
select a.game_id,
       a.team_id,
       b.local_date,
       a.outsPlayed as p_outsplayed,
       a.Strikeout as p_k,
       a.Walk as p_bb
from pitcher_counts a
join game_temp b
on a.game_id = b.game_id
where a.startingPitcher = 1
order by a.game_id);

/* organize table */
DROP TABLE IF EXISTS t3;
create table if not exists t3(
SELECT a.gameid,
       a.date,
       a.h_team,
       a.a_team,
       b.H as home_h,
       b.AB as home_ab,
       b.HR as home_hr,
       b.1B as home_1b,
       b.2B as home_2b,
       b.3B as home_3b,
       b.sf as home_sf,
       b.BB as home_bb,
       b.g as home_g,
       b.f as home_f,
       b.K as home_k,
       b.plate_appear as home_plate,
       b.hbp as home_hbp,
       c.p_outsplayed as home_p_o,
       c.p_k as home_p_k,
       c.p_bb as home_p_bb,
       d.H as away_h,
       d.AB as away_ab,
       d.HR as away_hr,
       d.1B as away_1b,
       d.2B as away_2b,
       d.3B as away_3b,
       d.sf as away_sf,
       d.K as away_k,
       d.BB as away_bb,
       d.g as away_g,
       d.f as away_f,
       d.hbp as away_hbp,
       d.plate_appear as away_plate,
       d.hbp,
       d.sf,
       e.p_outsplayed as away_p_o,
       e.p_k as away_p_k,
       e.p_bb as away_p_bb,
       f.finalScore as home_finalscore,
       g.finalScore as away_finalscore,
       h.home_pitch_substute as home_pitch_substute,
       h.away_pitch_substute as away_pitch_substute,
       i.streak as home_streak,
       j.streak as away_streak,
       k.temp as temp,
       a.hometeam_win
FROM t0 a
JOIN t1 b
ON a.h_team = b.team_id and a.gameid = b.game_id
JOIN p1 c
ON  a.h_team = c.team_id and a.gameid = c.game_id
JOIN t1 d
ON a.a_team = d.team_id and a.gameid = d.game_id
JOIN p1 e
ON  a.a_team = e.team_id and a.gameid = e.game_id
JOIN team_batting_counts f
ON a.h_team = f.team_id and a.gameid = f.game_id
JOIN team_batting_counts g
ON a.a_team = g.team_id and a.gameid = g.game_id
JOIN ps2 h
ON a.gameid = h.game_id
JOIN s1 i
ON a.h_team = i.team_id and a.gameid = i.game_id
JOIN s1 j
ON a.a_team = j.team_id and a.gameid = j.game_id
join w1 k
on a.gameid = k.game_id
order by a.gameid);

UPDATE t3
SET temp = REPLACE(temp, 'degrees', '')
WHERE temp LIKE '%degrees';


/* calcultate rolling100 */
DROP TABLE IF EXISTS result;
CREATE TABLE IF NOT EXISTS result(
select
        a.gameid,
        a.date,
        a.h_team,
        a.a_team,
        SUM(b.home_1b) as h_1b,
        SUM(b.home_2b) as h_2b,
        SUM(b.home_3b) as h_3b,
        SUM(b.home_ab) as h_ab,
        CASE WHEN SUM(b.home_hr) = 0 THEN 0
            else cast(SUM(b.home_ab) as float )/cast(SUM(b.home_hr) as float) end as h_at_bats_per_homerun,
        CASE WHEN SUM(b.home_h) = 0 THEN 0
            else cast(SUM(b.home_h) as float )/cast(SUM(b.home_ab) as float) end as h_batting_avg,
        SUM(b.home_bb) as h_bb,
        CASE WHEN SUM(b.home_h) = 0 THEN 0
            else cast(SUM(b.home_ab)-SUM(b.home_hr) as float )/cast(SUM(b.home_ab)-SUM(b.home_k)-SUM(b.home_hr)+SUM(b.home_sf) as float) end as h_babip,
        CASE WHEN SUM(b.home_k) = 0 THEN 0
            else cast(SUM(b.home_bb) as float )/cast(SUM(b.home_k) as float) end as h_walk2strikeout_ratio,
        CASE WHEN ((
                    1.4*(SUM(b.home_h)+SUM(b.home_2b)+(2*SUM(b.home_3b))+(3*SUM(b.home_hr))))
                    -0.6*SUM(b.home_h)-3*SUM(b.home_hr)+0.1*SUM(b.home_bb)*1.02)+(SUM(b.home_ab)-SUM(b.home_h)) = 0 THEN SUM(b.home_hr)
            else (cast(
                    (SUM(b.home_h)+SUM(b.home_bb)-SUM(b.home_hr)) *((
                    1.4*(SUM(b.home_h)+SUM(b.home_2b)+(2*SUM(b.home_3b))+(3*SUM(b.home_hr))))
                    -0.6*SUM(b.home_h)-3*SUM(b.home_hr)+0.1*SUM(b.home_bb)*1.02)
                    as float )/cast(((
                    1.4*(SUM(b.home_h)+SUM(b.home_2b)+(2*SUM(b.home_3b))+(3*SUM(b.home_hr))))
                    -0.6*SUM(b.home_h)-3*SUM(b.home_hr)+0.1*SUM(b.home_bb)*1.02)+(SUM(b.home_ab)-SUM(b.home_h)) as float))+SUM(b.home_hr) end as h_baserun,
        SUM(b.home_h)+SUM(b.home_2b)+(2*SUM(b.home_3b))+(3*SUM(b.home_hr)) as h_tb,
        CASE WHEN SUM(b.home_f) = 0 THEN 0
            else cast(SUM(b.home_g) as float )/cast(SUM(b.home_f) as float) end as h_groundfly_ratio,
        SUM(b.home_2b)+SUM(b.home_3b)+SUM(b.home_hr) as h_xbh,
        SUM(b.home_plate) as h_plate,
        SUM(b.home_k) as h_strikeout,
        CASE WHEN SUM(b.home_bb) = 0 THEN 0
        else cast(SUM(b.home_k) as float )/cast(SUM(b.home_bb) as float) end as h_strike_to_walk,
        CASE WHEN SUM(b.home_ab) = 0 THEN 0
            else cast((SUM(b.home_1b)+2*SUM(b.home_2b)+3*SUM(b.home_3b)+4*SUM(b.home_hr)) as float )/cast(SUM(b.home_ab) as float) end as h_slugging_avg,
        SUM(b.home_h)+SUM(b.home_bb)+SUM(b.home_hbp) as h_tob,
        SUM(b.home_sf) as h_sf,
        CASE WHEN SUM(b.home_ab) = 0 THEN 0
        else cast((SUM(b.home_1b)+2*SUM(b.home_2b)+3*SUM(b.home_3b)+4*SUM(b.home_hr)) as float )/cast(SUM(b.home_ab) as float) end as h_slugging_percentage,
        CASE WHEN SUM(b.home_ab)+SUM(b.home_bb)+SUM(b.home_hbp)+SUM(b.home_sf)= 0 THEN 0
        else cast(SUM(b.home_h)+SUM(b.home_bb)+SUM(b.home_hbp) as float )/cast(SUM(b.home_ab)+SUM(b.home_bb)+SUM(b.home_hbp)+SUM(b.home_sf) as float) end as h_on_base_percentage,
        CASE WHEN SUM(b.home_k) = 0 THEN 0
        else cast(SUM(b.home_plate) as float )/cast(SUM(b.home_k) as float) end as h_plate_per_strike,
        CASE WHEN SUM(b.home_ab) = 0 THEN 0
        else cast(SUM(b.home_2b)+SUM(b.home_3b)+SUM(b.home_hr) as float )/cast(SUM(b.home_ab) as float) end as h_iso,
        SUM(b.home_finalscore) as h_finalscore,
        CASE WHEN SUM(b.home_p_bb) = 0 THEN 0
        else cast(SUM(b.home_p_k) as float )/cast(SUM(b.home_p_bb) as float) end as h_p_strike_to_walk,
        cast(SUM(b.home_p_o) as float )/3 as h_p_ip,


        SUM(b.away_1b) as a_1b,
        SUM(b.away_2b) as a_2b,
        SUM(b.away_3b) as a_3b,
        SUM(b.away_ab) as a_ab,
        CASE WHEN SUM(b.away_hr) = 0 THEN 0
            else cast(SUM(b.away_ab) as float )/cast(SUM(b.away_hr) as float) end as a_at_bats_per_homerun,
        CASE WHEN SUM(b.away_h) = 0 THEN 0
            else cast(SUM(b.away_h) as float )/cast(SUM(b.away_ab) as float) end as a_batting_avg,
        SUM(b.away_bb) as a_bb,
        CASE WHEN SUM(b.away_ab)-SUM(b.away_k)-SUM(b.away_hr)+SUM(b.away_sf) = 0 THEN 0
            else cast(SUM(b.away_ab)-SUM(b.away_hr) as float )/cast(SUM(b.away_ab)-SUM(b.away_k)-SUM(b.away_hr)+SUM(b.away_sf) as float) end as a_babip,
        CASE WHEN SUM(b.away_k) = 0 THEN 0
            else cast(SUM(b.away_bb) as float )/cast(SUM(b.away_k) as float) end as a_walk2strikeout_ratio,
        CASE WHEN ((
                    1.4*(SUM(b.away_h)+SUM(b.away_2b)+(2*SUM(b.away_3b))+(3*SUM(b.away_hr))))
                    -0.6*SUM(b.away_h)-3*SUM(b.away_hr)+0.1*SUM(b.away_bb)*1.02)+(SUM(b.away_ab)-SUM(b.away_h)) = 0 THEN SUM(b.away_hr)
            else (cast(
                    (SUM(b.away_h)+SUM(b.away_bb)-SUM(b.away_hr)) *((
                    1.4*(SUM(b.away_h)+SUM(b.away_2b)+(2*SUM(b.away_3b))+(3*SUM(b.away_hr))))
                    -0.6*SUM(b.away_h)-3*SUM(b.away_hr)+0.1*SUM(b.away_bb)*1.02)
                    as float )/cast(((
                    1.4*(SUM(b.away_h)+SUM(b.away_2b)+(2*SUM(b.away_3b))+(3*SUM(b.away_hr))))
                    -0.6*SUM(b.away_h)-3*SUM(b.away_hr)+0.1*SUM(b.away_bb)*1.02)+(SUM(b.away_ab)-SUM(b.away_h)) as float))+SUM(b.away_hr) end as a_baserun,
        SUM(b.away_h)+SUM(b.away_2b)+(2*SUM(b.away_3b))+(3*SUM(b.away_hr)) as a_tb,
        CASE WHEN SUM(b.away_f) = 0 THEN 0
            else cast(SUM(b.away_g) as float )/cast(SUM(b.away_f) as float) end as a_groundfly_ratio,
        SUM(b.away_2b)+SUM(b.away_3b)+SUM(b.away_hr) as a_xbh,
        SUM(b.away_plate) as a_plate,
        SUM(b.away_k) as a_strikeout,
        CASE WHEN SUM(b.away_bb) = 0 THEN 0
        else cast(SUM(b.away_k) as float )/cast(SUM(b.away_bb) as float) end as a_strike_to_walk,
        CASE WHEN SUM(b.away_ab) = 0 THEN 0
           else cast((SUM(b.away_1b)+2*SUM(b.away_2b)+3*SUM(b.away_3b)+4*SUM(b.away_hr)) as float )/cast(SUM(b.away_ab) as float) end as a_slugging_avg,
        SUM(b.away_h)+SUM(b.away_bb)+SUM(b.away_hbp) as a_tob,
        SUM(b.away_sf) as a_sf,
        CASE WHEN SUM(b.away_ab) = 0 THEN 0
        else cast((SUM(b.away_1b)+2*SUM(b.away_2b)+3*SUM(b.away_3b)+4*SUM(b.away_hr)) as float )/cast(SUM(b.away_ab) as float) end as a_slugging_percentage,
        CASE WHEN SUM(b.away_ab)+SUM(b.away_bb)+SUM(b.away_hbp)+SUM(b.away_sf)= 0 THEN 0
        else cast(SUM(b.away_h)+SUM(b.away_bb)+SUM(b.away_hbp) as float )/cast(SUM(b.away_ab)+SUM(b.away_bb)+SUM(b.away_hbp)+SUM(b.away_sf) as float) end as a_on_base_percentage,
        CASE WHEN SUM(b.away_k) = 0 THEN 0
        else cast(SUM(b.away_plate) as float )/cast(SUM(b.away_k) as float) end as a_plate_per_strike,
        CASE WHEN SUM(b.away_ab) = 0 THEN 0
        else cast(SUM(b.away_2b)+SUM(b.away_3b)+SUM(b.away_hr) as float )/cast(SUM(b.away_ab) as float) end as a_iso,
        SUM(b.home_pitch_substute) as home_pitch_substute,
        SUM(b.away_pitch_substute) as away_pitch_substute,
        SUM(b.home_pitch_substute) - SUM(b.away_pitch_substute) as pitch_substract,
        (SUM(b.home_2b)+SUM(b.home_3b)+SUM(b.home_hr))-(SUM(b.away_2b)+SUM(b.away_3b)+SUM(b.away_hr)) as diff_xbh,
        SUM(b.away_finalscore) as a_finalscore,
        CASE WHEN SUM(b.away_p_bb) = 0 THEN 0
        else cast(SUM(b.away_p_k) as float )/cast(SUM(b.away_p_bb) as float) end as a_p_strike_to_walk,
        cast(SUM(b.away_p_o) as float )/3 as a_p_ip,
        a.home_streak,
        a.away_streak,
        avg(cast(b.temp as unsigned )) as avg_temp,
        a.temp as temp,
        a.hometeam_win
from t3 a
join t3 b
on a.h_team = b.h_team and a.a_team = b.a_team
where b.date between DATE_ADD(a.date, interval -7 day ) and DATE_ADD(a.date, interval -1 day)
group by a.gameid, a.date
order by a.date);

show tables;"

echo"done"

python /final.py

Exit
