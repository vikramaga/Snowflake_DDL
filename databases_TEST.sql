create or replace database TEST;

create or replace schema PUBLIC;

create or replace schema STG;

create or replace TABLE INSTRUMENT (
	C1 VARCHAR(16777216),
	C2 VARCHAR(16777216)
);
create or replace TABLE MYTABLE (
	AMOUNT NUMBER(38,0) autoincrement
);
create or replace TABLE TABLE_NEW (
	ID NUMBER(38,0) autoincrement,
	NAME VARCHAR(16777216)
);
create or replace TABLE TABLE_NEW1 (
	ID NUMBER(38,0) autoincrement,
	NAME VARCHAR(16777216)
);
create or replace TABLE TABLE_NEW2 (
	ID NUMBER(38,0) autoincrement,
	NAME VARCHAR(16777216)
);