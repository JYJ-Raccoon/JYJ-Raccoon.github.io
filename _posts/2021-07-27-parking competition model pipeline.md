---
layout: single
title:  "모델 파이프라인"
---

## 패키지


```python
# 기본 패키지
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
# 모델 검증
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# 모델 종류
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor

```

## 데이터


```python
train = pd.read_csv('../Data/train.csv')
test = pd.read_csv('../Data/test.csv')
submission = pd.read_csv('../Data/sample_submission.csv')
```


```python
train.shape, test.shape
```




    ((2952, 15), (1022, 14))




```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>단지코드</th>
      <th>총세대수</th>
      <th>임대건물구분</th>
      <th>지역</th>
      <th>공급유형</th>
      <th>전용면적</th>
      <th>전용면적별세대수</th>
      <th>공가수</th>
      <th>자격유형</th>
      <th>임대보증금</th>
      <th>임대료</th>
      <th>도보 10분거리 내 지하철역 수(환승노선 수 반영)</th>
      <th>도보 10분거리 내 버스정류장 수</th>
      <th>단지내주차면수</th>
      <th>등록차량수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C2483</td>
      <td>900</td>
      <td>아파트</td>
      <td>경상북도</td>
      <td>국민임대</td>
      <td>39.72</td>
      <td>134</td>
      <td>38.0</td>
      <td>A</td>
      <td>15667000</td>
      <td>103680</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1425.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C2483</td>
      <td>900</td>
      <td>아파트</td>
      <td>경상북도</td>
      <td>국민임대</td>
      <td>39.72</td>
      <td>15</td>
      <td>38.0</td>
      <td>A</td>
      <td>15667000</td>
      <td>103680</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1425.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C2483</td>
      <td>900</td>
      <td>아파트</td>
      <td>경상북도</td>
      <td>국민임대</td>
      <td>51.93</td>
      <td>385</td>
      <td>38.0</td>
      <td>A</td>
      <td>27304000</td>
      <td>184330</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1425.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C2483</td>
      <td>900</td>
      <td>아파트</td>
      <td>경상북도</td>
      <td>국민임대</td>
      <td>51.93</td>
      <td>15</td>
      <td>38.0</td>
      <td>A</td>
      <td>27304000</td>
      <td>184330</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1425.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C2483</td>
      <td>900</td>
      <td>아파트</td>
      <td>경상북도</td>
      <td>국민임대</td>
      <td>51.93</td>
      <td>41</td>
      <td>38.0</td>
      <td>A</td>
      <td>27304000</td>
      <td>184330</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1425.0</td>
      <td>1015.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>단지코드</th>
      <th>총세대수</th>
      <th>임대건물구분</th>
      <th>지역</th>
      <th>공급유형</th>
      <th>전용면적</th>
      <th>전용면적별세대수</th>
      <th>공가수</th>
      <th>자격유형</th>
      <th>임대보증금</th>
      <th>임대료</th>
      <th>도보 10분거리 내 지하철역 수(환승노선 수 반영)</th>
      <th>도보 10분거리 내 버스정류장 수</th>
      <th>단지내주차면수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C1072</td>
      <td>754</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>39.79</td>
      <td>116</td>
      <td>14.0</td>
      <td>H</td>
      <td>22830000</td>
      <td>189840</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>683.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C1072</td>
      <td>754</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>46.81</td>
      <td>30</td>
      <td>14.0</td>
      <td>A</td>
      <td>36048000</td>
      <td>249930</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>683.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C1072</td>
      <td>754</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>46.90</td>
      <td>112</td>
      <td>14.0</td>
      <td>H</td>
      <td>36048000</td>
      <td>249930</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>683.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C1072</td>
      <td>754</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>46.90</td>
      <td>120</td>
      <td>14.0</td>
      <td>H</td>
      <td>36048000</td>
      <td>249930</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>683.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C1072</td>
      <td>754</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>51.46</td>
      <td>60</td>
      <td>14.0</td>
      <td>H</td>
      <td>43497000</td>
      <td>296780</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>683.0</td>
    </tr>
  </tbody>
</table>
</div>



## 결측치 확인


```python
train.isna().sum()
```




    단지코드                              0
    총세대수                              0
    임대건물구분                            0
    지역                                0
    공급유형                              0
    전용면적                              0
    전용면적별세대수                          0
    공가수                               0
    자격유형                              0
    임대보증금                           569
    임대료                             569
    도보 10분거리 내 지하철역 수(환승노선 수 반영)    211
    도보 10분거리 내 버스정류장 수                4
    단지내주차면수                           0
    등록차량수                             0
    dtype: int64




```python
test.isna().sum()
```




    단지코드                              0
    총세대수                              0
    임대건물구분                            0
    지역                                0
    공급유형                              0
    전용면적                              0
    전용면적별세대수                          0
    공가수                               0
    자격유형                              2
    임대보증금                           180
    임대료                             180
    도보 10분거리 내 지하철역 수(환승노선 수 반영)     42
    도보 10분거리 내 버스정류장 수                0
    단지내주차면수                           0
    dtype: int64



## 컬럼명 변경

지하쳘역 수와 버스 정류장 수의 컬럼명을 지하철, 버스로 변경하였습니다.


```python
train.columns
```




    Index(['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
           '자격유형', '임대보증금', '임대료', '도보 10분거리 내 지하철역 수(환승노선 수 반영)',
           '도보 10분거리 내 버스정류장 수', '단지내주차면수', '등록차량수'],
          dtype='object')




```python
train.columns = [
    '단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수', '신분',
    '임대보증금', '임대료', '지하철', '버스',
    '단지내주차면수', '등록차량수'
]

test.columns = [
    '단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수', '신분',
    '임대보증금', '임대료', '지하철', '버스',
    '단지내주차면수'
]
```

## 지역명 숫자로 매핑


```python
local_map = {}
for i, loc in enumerate(train['지역'].unique()):
    local_map[loc] = i
```


```python
train['지역'] = train['지역'].map(local_map)
test['지역'] = test['지역'].map(local_map)
```

## 전용면적을 5의 배수로 변경


```python
train['전용면적'] = train['전용면적']//5*5
test['전용면적'] = test['전용면적']//5*5
```

## 전용면적 상/하한 적용

상한100, 하한 15


```python
idx = train[train['전용면적']>100].index
train.loc[idx, '전용면적'] = 100
idx = test[test['전용면적']>100].index
test.loc[idx, '전용면적'] = 100

idx = train[train['전용면적']<15].index
train.loc[idx, '전용면적'] = 15
idx = test[test['전용면적']<15].index
test.loc[idx, '전용면적'] = 15
```


```python
test['전용면적'].unique()
```




    array([ 35.,  45.,  50.,  30.,  55.,  25.,  75., 100.,  15.,  20.,  40.,
            60.,  80.,  70.])



## 단지별 데이터 1차원으로 취합


```python
columns = ['단지코드', '총세대수', '공가수', '지역', '단지내주차면수', '지하철', '버스']
target = '등록차량수'
area_columns = []
for area in train['전용면적'].unique():
    area_columns.append(f'면적_{area}')
```


```python
new_train = pd.DataFrame()
new_test = pd.DataFrame()
```


```python
for i, code in tqdm(enumerate(train['단지코드'].unique())):
    temp = train[train['단지코드']==code]
    temp.index = range(temp.shape[0])
    for col in columns:
        new_train.loc[i, col] = temp.loc[0, col]
    
    for col in area_columns:
        area = float(col.split('_')[-1])
        new_train.loc[i, col] = temp[temp['전용면적']==area]['전용면적별세대수'].sum()
    
    new_train.loc[i, '등록차량수'] = temp.loc[0, '등록차량수']
    
for i, code in tqdm(enumerate(test['단지코드'].unique())):
    temp = test[test['단지코드']==code]
    temp.index = range(temp.shape[0])
    for col in columns:
        new_test.loc[i, col] = temp.loc[0, col]
    
    for col in area_columns:
        area = float(col.split('_')[-1])
        new_test.loc[i, col] = temp[temp['전용면적']==area]['전용면적별세대수'].sum()
```

    423it [00:05, 80.24it/s]
    150it [00:01, 81.03it/s]
    

## 결측치 처리


```python
new_train = new_train.fillna(-1)
new_test = new_test.fillna(-1)
```

## 스케일링


```python
del new_train['단지코드']
del new_test['단지코드']
```


```python
new_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>총세대수</th>
      <th>공가수</th>
      <th>지역</th>
      <th>단지내주차면수</th>
      <th>지하철</th>
      <th>버스</th>
      <th>면적_35.0</th>
      <th>면적_50.0</th>
      <th>면적_55.0</th>
      <th>면적_30.0</th>
      <th>...</th>
      <th>면적_25.0</th>
      <th>면적_70.0</th>
      <th>면적_15.0</th>
      <th>면적_20.0</th>
      <th>면적_100.0</th>
      <th>면적_60.0</th>
      <th>면적_75.0</th>
      <th>면적_80.0</th>
      <th>면적_65.0</th>
      <th>등록차량수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>900.0</td>
      <td>38.0</td>
      <td>0.0</td>
      <td>1425.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>149.0</td>
      <td>665.0</td>
      <td>86.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>545.0</td>
      <td>17.0</td>
      <td>1.0</td>
      <td>624.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>80.0</td>
      <td>132.0</td>
      <td>0.0</td>
      <td>276.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>205.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1216.0</td>
      <td>13.0</td>
      <td>2.0</td>
      <td>1285.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>124.0</td>
      <td>0.0</td>
      <td>390.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1064.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>755.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>734.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>240.0</td>
      <td>303.0</td>
      <td>212.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>730.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>696.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>645.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>254.0</td>
      <td>246.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>553.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>418</th>
      <td>90.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>66.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>42.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>419</th>
      <td>492.0</td>
      <td>24.0</td>
      <td>5.0</td>
      <td>521.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>156.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>156.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>246.0</td>
    </tr>
    <tr>
      <th>420</th>
      <td>40.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>421</th>
      <td>90.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>422</th>
      <td>239.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>166.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>201.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>146.0</td>
    </tr>
  </tbody>
</table>
<p>423 rows × 22 columns</p>
</div>




```python
new_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>총세대수</th>
      <th>공가수</th>
      <th>지역</th>
      <th>단지내주차면수</th>
      <th>지하철</th>
      <th>버스</th>
      <th>면적_35.0</th>
      <th>면적_50.0</th>
      <th>면적_55.0</th>
      <th>면적_30.0</th>
      <th>...</th>
      <th>면적_40.0</th>
      <th>면적_25.0</th>
      <th>면적_70.0</th>
      <th>면적_15.0</th>
      <th>면적_20.0</th>
      <th>면적_100.0</th>
      <th>면적_60.0</th>
      <th>면적_75.0</th>
      <th>면적_80.0</th>
      <th>면적_65.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>754.0</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>683.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>116.0</td>
      <td>376.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1354.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>1216.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>420.0</td>
      <td>578.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>619.0</td>
      <td>18.0</td>
      <td>8.0</td>
      <td>547.0</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>165.0</td>
      <td>132.0</td>
      <td>44.0</td>
      <td>82.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>593.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>543.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>335.0</td>
      <td>84.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1297.0</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>1112.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>425.0</td>
      <td>304.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>349.0</td>
      <td>17.0</td>
      <td>9.0</td>
      <td>270.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>138.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>112.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>146</th>
      <td>596.0</td>
      <td>35.0</td>
      <td>11.0</td>
      <td>593.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>304.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>180.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>147</th>
      <td>120.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>54.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>675.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>467.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>126.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>74.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>240.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149</th>
      <td>382.0</td>
      <td>45.0</td>
      <td>4.0</td>
      <td>300.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>202.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>116.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 21 columns</p>
</div>



## 데이터셋 분할


```python
x_train = new_train.iloc[:, 1:-1]
y_train = new_train.iloc[:,-1]
x_test = new_test.iloc[:,1:]
```


```python
from sklearn.preprocessing import PowerTransformer

# 스케일링 - 정규분포에 가깝게 만들기
power = PowerTransformer(standardize = True)
x_train = power.fit_transform(x_train)
x_test = power.transform(x_test)
```

## 검증


```python
models = [RandomForestRegressor(n_jobs=-1, random_state=42), 
          Ridge(random_state=42), Lasso(random_state=42),ElasticNet(random_state=42),ARDRegression(),
          BayesianRidge(),XGBRegressor(random_state=42),LGBMRegressor(random_state=42),
          GradientBoostingRegressor(random_state=42),
          CatBoostRegressor(random_state=42)
         ]
score = []
std = []
for model in models:
    kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)
    scores = cross_val_score(model,
                         x_train,
                         y_train,
                         scoring='neg_mean_squared_error',
                         cv = kfold,
                         n_jobs = -1)
    scores *= -1
    score.append(scores.mean())
    std.append(scores.std())
    print("{}\n[Mean]: {:.3f} [Std]: {:.3f}\n".format(
        '<' + str(model).split('(')[0] + '>', scores.mean(), scores.std(), scores.min(), scores.max()))
```

    <RandomForestRegressor>
    [Mean]: 37793.216 [Std]: 4065.930
    
    <Ridge>
    [Mean]: 42758.225 [Std]: 7841.175
    
    <Lasso>
    [Mean]: 42593.955 [Std]: 7791.982
    
    <ElasticNet>
    [Mean]: 54702.318 [Std]: 10844.682
    
    <ARDRegression>
    [Mean]: 42501.962 [Std]: 7610.200
    
    <BayesianRidge>
    [Mean]: 42793.855 [Std]: 8197.349
    
    <XGBRegressor>
    [Mean]: 40278.315 [Std]: 6030.343
    
    <LGBMRegressor>
    [Mean]: 37360.517 [Std]: 6305.987
    
    <GradientBoostingRegressor>
    [Mean]: 36435.335 [Std]: 5594.649
    
    <<catboost.core.CatBoostRegressor object at 0x0000022E7843A280>>
    [Mean]: 35801.707 [Std]: 3702.012
    
    


```python
# 평균과 표준편차 모두 낮아야 일반화됨
def mean_std_plot(co1, co2, data):    
    plt.figure(figsize = (20,10))
    g = sns.scatterplot(x = 'score', y = 'std', data = df, s = 80, color = 'red', )
    for line in range(df.shape[0]):
        g.text(x = df.score[line]-300, y = df['std'][line]+100, s = df.model[line], fontsize = 15)
    plt.grid()
    plt.ylabel('Std')
    plt.xlabel('Score')
    plt.show()
```


```python
# 모델이름 정리
model_plot = models.copy()
for index,i in enumerate(model_plot):
    if str(i).startswith('<') :
        model_plot[index] = 'CatboostRegressor'
    else : model_plot[index] = str(i).split('(')[0]

# 데이터프레임으로 변환        
df = pd.DataFrame({'model' : model_plot, 'score' : score, 'std' : std})
# 평균과 표준편차 그래프 그리기 (좌하단일수록 좋음)
mean_std_plot('score','std', df)
```


    
![png](output_37_0.png)
    


## 학습


```python
# 평균이 최소인 모델로 학습 (std도 고려할 필요가 있음)
model = models[score.index(min(score))]
print(f'학습할 모델 : {model}')

try :
    model.fit(x_train, y_train, verbose = 0)
except :
    model.fit(x_train, y_train)
```

    학습할 모델 : <catboost.core.CatBoostRegressor object at 0x0000022E7843A280>
    

## 추론 및 제출


```python
pred = model.predict(x_test)
```


```python
submission['num'] = pred
```


```python
submission
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C1072</td>
      <td>783.003773</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C1128</td>
      <td>1291.399676</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C1456</td>
      <td>484.426734</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C1840</td>
      <td>530.692542</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C1332</td>
      <td>1131.322119</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>C2456</td>
      <td>242.422248</td>
    </tr>
    <tr>
      <th>146</th>
      <td>C1266</td>
      <td>490.560389</td>
    </tr>
    <tr>
      <th>147</th>
      <td>C2152</td>
      <td>37.012084</td>
    </tr>
    <tr>
      <th>148</th>
      <td>C1267</td>
      <td>334.011444</td>
    </tr>
    <tr>
      <th>149</th>
      <td>C2189</td>
      <td>156.765811</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 2 columns</p>
</div>



### Make Submissions


```python
t = pd.Timestamp.now()
fname = f'submissions_dacon_{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}.csv'
submission.to_csv(fname, index=False)
print("'{}' is ready to submit." .format(fname))
```

    'submissions_dacon_07121645.csv' is ready to submit.
    

### END
