# Files

http://essay.utwente.nl/61317/1/MSc_CZ_Michorius.pdf 

OPTIONS COMPRESS=YES;
OPTIONS REPLACE;

LIBNAME WF 'D:\Zhihui\LapseRatio';

%let filedate = %sysfunc(putn(%eval(%sysfunc(today())-1),yymmddn8.));
%PUT &filedate;

PROC SQL;
CREATE TABLE WF.DISC_RATE AS
SELECT
POLICY_ID,
ITEM_ID,
ROUND(NEXT_DISCNT_PREM_AF/NEXT_STD_PREM_AF,0.01) AS NEXT_PREM_DISC_RATE,
NEXT_DISCNT_PREM_AF/NEXT_STD_PREM_AF as NEXT_PREM_DISC_RATE_Ori
FROM DS.S938166_T_CONTRCT_PRODUCT;
QUIT;

%macro SHP(timeid);

PROC SQL;

CREATE TABLE work.SHP_&timeid AS

SELECT 
A.SRC_SYS,
A.BIZ_DT,
A.POL_KEY,
B.BEN_KEY,
&timeid AS TIME_ID,
ifc(&timeid=1,'Actual','Projected') as Data_Status,
ifn(&timeid=1,ifn(B.BEN_NEXT_PREM_DUE_DT>='01NOV2015'D,1,0),ifn(B.BEN_NEXT_PREM_DUE_DT>='01NOV2015'D,1,.)) as SHP_CT,
ifn(&timeid=1,
	ifn(A.POL_COV_STATUS='INF',intck('year',B.BEN_COMMENCE_DT,intnx('day',today(),0),'C'),intck('year',A.POL_ISSUE_DT,intnx('day',A.POL_LAPSE_DT,0),'C')),
	.) AS C_PolicyYear,
A.POL_NO,
A.POL_ISSUE_DT,
A.POL_LAPSE_DT,
A.POL_HLDR_KEY,
A.POL_ISSUE_AGT_KEY,
A.POL_SERV_AGT_KEY,
	J.AGT_MGR_NO,
	J.AGT_DIR_NO,
	J.AGT_CLUSTER_CD,
(SELECT BMAP_NAME
FROM DS.BMAP
WHERE SRC_SYS=A.SRC_SYS
AND BMAP_VAL=A.POL_PREM_PAY_MODE
AND BMAP_COL='PREM_PAY_MODE') AS PREM_PAY_MODE,

(SELECT BMAP_NAME
FROM DS.BMAP
WHERE SRC_SYS=A.SRC_SYS
AND BMAP_VAL=A.POL_PREM_FREQ
AND BMAP_COL='PREM_FREQ') AS PREM_FREQ,

A.POL_COV_STATUS AS POL_STATUS,
B.BEN_LIFE_ASD_1_ENTRY_AGE AS ENTRY_AGE,
B.BEN_COMMENCE_DT AS COMMENCE_DT,
intnx('year',B.BEN_NEXT_PREM_DUE_DT,&timeid-1,"sameday") AS NEXT_PREM_DUE_DT format=YYMMDDD10.,
INTCK('YEAR',B.BEN_COMMENCE_DT,B.BEN_NEXT_PREM_DUE_DT) + &timeid-1  AS POL_YR_NEXT_PREM_DUE_DT,

B.BEN_LIFE_ASD_1_ENTRY_AGE+
INTCK('YEAR',B.BEN_COMMENCE_DT,B.BEN_NEXT_PREM_DUE_DT) + &timeid-1 AS AGE_NEXT_PREM_DUE_DT,

B.BEN_PROD_CD AS PROD_CD,

C.BASIC_PREM_AMT AS NEXT_BASIC_PREM_AMT,
ROUND(C.BASIC_PREM_AMT * G.NEXT_PREM_DISC_RATE_Ori,0.01) AS NEXT_PREM_DISC_AMT,
G.NEXT_PREM_DISC_RATE,
ROUND(CASE WHEN F.GST_INDICATOR='1' THEN 0.07*(C.BASIC_PREM_AMT - C.BASIC_PREM_AMT * G.NEXT_PREM_DISC_RATE_Ori) 
	ELSE 0 END,0.01) AS NEXT_GST_AMT,
C.BASIC_PREM_AMT - C.BASIC_PREM_AMT * G.NEXT_PREM_DISC_RATE_Ori + CALCULATED NEXT_GST_AMT AS NEXT_INST_PREM_AMT,
C.BASIC_PREM_AMT - C.BASIC_PREM_AMT * G.NEXT_PREM_DISC_RATE_Ori AS NEXT_WGHT_PREM_AMT,

CASE 
  WHEN CALCULATED AGE_NEXT_PREM_DUE_DT BETWEEN 1 AND 40 THEN 300
  WHEN CALCULATED AGE_NEXT_PREM_DUE_DT BETWEEN 41 AND 70 THEN 600
  ELSE 900
END AS AWL_AMT,

C.CASH_INST,

CASE
  WHEN CALCULATED NEXT_INST_PREM_AMT>CALCULATED AWL_AMT THEN CALCULATED NEXT_INST_PREM_AMT-CALCULATED AWL_AMT
  ELSE 0
END AS CASH_AMT

FROM DS.POLICY AS A

INNER JOIN DS.BENEFIT AS B
ON A.SRC_SYS=B.SRC_SYS
AND A.BIZ_DT=B.BIZ_DT
AND A.POL_KEY=B.POL_KEY
AND A.SRC_SYS='FPM'
AND A.POL_COV_STATUS IN ('INF','LAP')
AND B.BEN_PROD_CD IN ('M19A4','M19B4','M19S4','M27A3','M27B3','M27P3')

INNER JOIN WF.SHP_PREM AS C
ON B.BEN_PROD_CD=C.PROD_CD
AND B.BEN_LIFE_ASD_1_ENTRY_AGE+
INTCK('YEAR',B.BEN_COMMENCE_DT,B.BEN_NEXT_PREM_DUE_DT) + &timeid-1=C.AGE_NEXT_BIRTHDAY

INNER JOIN DS.S938156_T_CONTRACT_MASTER AS D
ON A.BIZ_DT=D.BUSINESSDATE
AND A.POL_KEY=D.POLICY_ID
AND D._SUSPEND='N'

INNER JOIN DS.S938141_T_PRODUCT_LIFE AS F
ON B.BIZ_DT=F.BUSINESSDATE
AND B.BEN_PROD_CD=F.INTERNAL_ID

INNER JOIN WF.DISC_rate AS G
ON B.POL_KEY=G.POLICY_ID
AND B.BEN_KEY=G.ITEM_ID

	LEFT JOIN DS.agent AS J
	ON A.SRC_SYS=J.SRC_SYS
	AND A.BIZ_DT=J.BIZ_DT
	AND A.POL_SERV_AGT_KEY=J.AGT_KEY

where CALCULATED SHP_CT IS NOT MISSING;
/*WHERE B.BEN_NEXT_PREM_DUE_DT>='01NOV2015'D;*/

QUIT;

%mend;

%macro Loop;
%do i=1 %to 5;
%SHP(&i);

    %if &i>1 %then %do;
    proc sql;
    create table work.Temp as
    select A.*
	FROM WF.SHP_&filedate as A
	UNION
	select B.*
    FROM work.SHP_&i as B;
    quit;
    %end;
    %else %do;
    data work.Temp;
    set work.Shp_&i;
    run;
    %end;

data WF.SHP_&filedate;
SET work.Temp;
run;
%end;

%mend;

%Loop;

 
PROC EXPORT DATA=WF.SHP_&filedate
   outfile="D:\Zhihui\LapseRatio\shp.CSV"
   dbms=CSV
   replace;
RUN;

