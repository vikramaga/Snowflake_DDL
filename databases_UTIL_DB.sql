create or replace database UTIL_DB;

create or replace schema PUBLIC;

CREATE OR REPLACE FILE FORMAT CSV
	NULL_IF = ('NULL', 'null', '\\N')
COMMENT='parse comma-delimited data'
;
CREATE OR REPLACE FILE FORMAT CSV_DQ
	FIELD_OPTIONALLY_ENCLOSED_BY = '\"'
	NULL_IF = ('NULL', 'null', '\\N')
COMMENT='parse comma-delimited, double-quoted data'
;
CREATE OR REPLACE FILE FORMAT JSON
	TYPE = JSON
COMMENT='parse JSON data'
;
CREATE OR REPLACE FILE FORMAT PARQUET
	TYPE = PARQUET
	COMPRESSION = NONE
COMMENT='parse uncompressed Parquet data'
;
CREATE OR REPLACE FILE FORMAT PSV
	FIELD_DELIMITER = '|'
	NULL_IF = ('NULL', 'null', '\\N')
COMMENT='parse pipe-delimited data'
;
CREATE OR REPLACE FILE FORMAT PSV_DQ
	FIELD_DELIMITER = '|'
	FIELD_OPTIONALLY_ENCLOSED_BY = '\"'
	NULL_IF = ('NULL', 'null', '\\N')
COMMENT='parse pipe-delimited, double-quoted data'
;
CREATE OR REPLACE FILE FORMAT SINGLE_COLUMN_ROWS
	FIELD_DELIMITER = 'NONE'
	NULL_IF = ('')
COMMENT='copy each line into single-column row'
;
CREATE OR REPLACE FILE FORMAT SOH
	FIELD_DELIMITER = '\u0001'
	NULL_IF = ('NULL', 'null', '\\N')
COMMENT='parse SOH-delimited (0x01) data'
;
CREATE OR REPLACE FILE FORMAT SOH_DQ
	FIELD_DELIMITER = '\u0001'
	FIELD_OPTIONALLY_ENCLOSED_BY = '\"'
	NULL_IF = ('NULL', 'null', '\\N')
COMMENT='parse SOH-delimited (0x01), double-quoted data'
;
CREATE OR REPLACE FILE FORMAT THORN
	FIELD_DELIMITER = '\u00FE'
	NULL_IF = ('NULL', 'null', '\\N')
COMMENT='parse thorn-delimited (0xFE) data'
;
CREATE OR REPLACE FILE FORMAT TSV
	FIELD_DELIMITER = '\t'
	NULL_IF = ('NULL', 'null', '\\N')
COMMENT='parse tab-delimited data'
;
CREATE OR REPLACE FILE FORMAT TSV_DQ
	FIELD_DELIMITER = '\t'
	FIELD_OPTIONALLY_ENCLOSED_BY = '\"'
	NULL_IF = ('NULL', 'null', '\\N')
COMMENT='parse tab-delimited, double-quoted data'
;
CREATE OR REPLACE FUNCTION "DECODE_URI"(URI VARCHAR)
RETURNS VARCHAR(16777216)
LANGUAGE JAVASCRIPT
AS '
var dec = decodeURIComponent(URI);
return dec ;
  ';
CREATE OR REPLACE FUNCTION "SFWHO"()
RETURNS TABLE (TS TIMESTAMP_LTZ(9), ACCOUNT VARCHAR(16777216), USER VARCHAR(16777216), ROLE VARCHAR(16777216), DATABASE VARCHAR(16777216), SCHEMA VARCHAR(16777216), WAREHOUSE VARCHAR(16777216))
LANGUAGE SQL
AS 'select
  current_timestamp(),
  current_account(),
  current_user(),
  current_role(),
  current_database(),
  current_schema(),
  current_warehouse()
  ';