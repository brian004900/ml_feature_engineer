create table historical_avg(
select
      game_id
    , batter
,case when atBat = 0 then 0
    else cast(SUM(Hit) as float )/cast(SUM(atBat) as float) end as batting_average
from batter_counts
group by batter
order by batter);


create table annual_batting_avg(
select battersInGame.game_id, battersInGame.batter, EXTRACT(year from game.local_date) as year
, case when batter_counts.atBat = 0 then 0
    else cast(SUM(batter_counts.Hit) as float )/cast(SUM(batter_counts.atBat) as float) end as batting_average
from battersInGame
join game
on battersInGame.game_id = game.game_id
join batter_counts
on battersInGame.batter = batter_counts.batter and battersInGame.game_id = batter_counts.game_id
group by year, battersInGame.batter
order by battersInGame.batter, year);


create temporary table t1 (
select battersInGame.game_id
, battersInGame.batter, local_date
, batter_counts.Hit as hit, batter_counts.atBat as atbat
from battersInGame
join game_temp
on battersInGame.game_id = game_temp.game_id
join batter_counts
on battersInGame.batter = batter_counts.batter and battersInGame.game_id = batter_counts.game_id
order by battersInGame.batter);


create unique index t1_idx
on t1 (game_id, batter, local_date, hit, atbat);


create table rolling_batting_avg(
select
      a.game_id
    , a.batter
    , a.local_date
    , case when SUM(b.atbat) = 0 then 0
    else cast(SUM(b.hit) as float )/cast(SUM(b.atbat) as float) end as rolling_batting
from t1 a
join t1 b
on b.batter = a.batter
where b.local_date between DATE_ADD(a.local_date, interval -100 day ) and DATE_ADD(a.local_date, interval -1 day)
group by a.game_id, a.batter, a.local_date, a.hit, a.atbat
order by batter, a.local_date);