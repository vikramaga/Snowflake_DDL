create or replace database TEST;

create or replace schema PUBLIC;

CREATE OR REPLACE FUNCTION "ADDITION"(A NUMBER, B NUMBER)
RETURNS NUMBER(38,0)
LANGUAGE SQL
COMMENT='addition of two numbers'
AS 'a+b';
CREATE OR REPLACE FUNCTION "DEVIDE"(A NUMBER, B NUMBER)
RETURNS NUMBER(38,0)
LANGUAGE SQL
COMMENT='addition of two numbers'
AS 'a/b';
CREATE OR REPLACE FUNCTION "SUBSTRACTION"(A NUMBER, B NUMBER)
RETURNS NUMBER(38,0)
LANGUAGE SQL
COMMENT='substraction of two numbers'
AS 'a-b';
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