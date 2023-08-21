use baseball;

DROP TABLE IF EXISTS win_lose;
create temporary table win_lose(
SELECT game_id,
       team_id,
       local_date,
       home_away,
       win_lose,
       CASE WHEN home_away = 'H' and win_lose = 'W' THEN 1
            ELSE 0 END as win_or_lose
FROM team_results
order by game_id, team_id);


DROP TABLE IF EXISTS t0;
create temporary table t0(
select a.game_id as gameid,
       a.local_date as date,
       a.team_id as h_team,
       b.team_id as a_team,
       a.win_or_lose as hometeam_win
from win_lose a
join win_lose b
on a.game_id = b.game_id
where a.team_id != b.team_id and a.team_id = 5621
group by a.game_id
order by a.game_id);


DROP TABLE IF EXISTS t1;
create temporary table t1(
SELECT game_id,
       team_id,
    SUM(Hit) as H,
    SUM(atBat) as AB,
    SUM(Walk) as BB,
    SUM(Hit_By_Pitch) as HBP,
    SUM(Sac_Fly) as SF,
    SUM(Home_Run) as HR,
    SUM(Strikeout) as K,
    SUM(Intent_Walk) as IW,
    SUM(plateApperance) as plate_appear,
    SUM(Single) as 1B,
    SUM(`Double`) as 2B,
    SUM(Triple) as 3B
FROM batter_counts
GROUP BY game_id, team_id);

DROP TABLE IF EXISTS p1;
create temporary table p1(
select a.game_id,
       a.team_id,
       b.local_date,
       a.outsPlayed,
       a.Strikeout,
       a.Walk
from pitcher_counts a
join game_temp b
on a.game_id = b.game_id
where a.startingPitcher = 1
order by a.game_id);

DROP TABLE IF EXISTS t3;
create temporary table t3(
SELECT a.gameid,
       a.date,
       b.H as home_h,
       b.AB as home_ab,
       b.HR as home_hr,
       b.1B as home_1b,
       b.2B as home_2b,
       b.3B as home_3b,
       c.outsPlayed as home_o,
       c.Strikeout as home_k,
       c.Walk as home_bb,
       d.H as away_h,
       d.AB as away_ab,
       d.HR as away_hr,
       d.1B as away_1b,
       d.2B as away_2b,
       d.3B as away_3b,
       e.outsPlayed as away_o,
       e.Strikeout as away_k,
       e.Walk as away_bb,
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
order by a.gameid);



DROP TABLE IF EXISTS result;
CREATE TABLE result(
select
        a.gameid,
        a.date,
        CASE WHEN SUM(b.home_ab) = 0 THEN 0
            else cast(SUM(b.home_h) as float )/cast(SUM(b.home_ab) as float) end as h_batting_avg,
        CASE WHEN SUM(b.home_hr) = 0 THEN 0
            else cast(SUM(b.home_hr) as float )/cast(SUM(b.home_h) as float) end as h_home_run_per_hit,
        CASE WHEN SUM(b.home_ab) = 0 THEN 0
            else cast((SUM(b.home_1b)+2*SUM(b.home_2b)+3*SUM(b.home_3b)+4*SUM(b.home_hr)) as float )/cast(SUM(b.home_ab) as float) end as h_slugging_avg,
        cast(SUM(b.home_o) as float )/3 as h_ip,
        CASE WHEN SUM(b.home_bb) = 0 THEN 0
            else cast(SUM(b.home_k) as float )/cast(SUM(b.home_bb) as float) end as h_strike_to_walk,
        CASE WHEN SUM(b.away_ab) = 0 THEN 0
            else cast(SUM(b.away_h) as float )/cast(SUM(b.away_ab) as float) end as a_batting_avg,
        CASE WHEN SUM(b.away_hr) = 0 THEN 0
            else cast(SUM(b.away_hr) as float )/cast(SUM(b.away_h) as float) end as a_home_run_per_hit,
        CASE WHEN SUM(b.away_ab) = 0 THEN 0
            else cast((SUM(b.away_1b)+2*SUM(b.away_2b)+3*SUM(b.away_3b)+4*SUM(b.away_hr)) as float )/cast(SUM(b.away_ab) as float) end as a_slugging_avg,
        cast(SUM(b.away_o) as float )/3 as a_ip,
        CASE WHEN SUM(b.away_bb) = 0 THEN 0
        else cast(SUM(b.away_k) as float )/cast(SUM(b.away_bb) as float) end as a_strike_to_walk,
        a.hometeam_win
from t3 a
join t3 b
where b.date between DATE_ADD(a.date, interval -100 day ) and DATE_ADD(a.date, interval -1 day)
group by a.gameid, a.date
order by a.date);