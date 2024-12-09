import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import pickle
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Annotated, Union
from io import StringIO
from fastapi.responses import StreamingResponse
from io import BytesIO


app = FastAPI()

class Item(BaseModel):
    name: str
    year: Annotated[int, Field(strict=True, gt=0)]
    selling_price: Annotated[int, Field(strict=True, gt=0)]
    km_driven: Annotated[int, Field(strict=True, gt=0)]
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: Annotated[float, Field(strict=True, gt=0)]

def get_data(sample):
    try:
        item = Item.model_validate(sample)
    except Exception:
        print("Ошибка валидации входных данных")
        # print(type(sample))
        # print(sample)
    dt = sample.dict()
    for key in dt:
        dt[key] = [dt[key]]

    return pd.DataFrame(dt)

def get_data_lst(items):
    lst = []
    for i in range(len(items)):
        items[i] = items[i].dict()
        for key in items[i]:
            items[i][key] = [items[i][key]]
        lst.append(pd.DataFrame(items[i]))

    return pd.concat(lst)
    

# удаление дубликатов. Если при одинаковом признаковом описании цены на автомобили отличаются, 
# то оставим первую строку по этому автомобилю
def del_duplicates(df_train : pd.DataFrame) -> pd.DataFrame:

    def foo(X):
        X = X.fillna(0) # чтобы находить идентичные строки
        lst_unique = []
        for i in tqdm(range(X.drop_duplicates().shape[0])):
            elem = X.iloc[i]
            sub = X[X[list(elem.keys())] == pd.Series(elem)].dropna()
            sub = pd.DataFrame(sub.iloc[0]).T
            if sub.shape[0] == 1:
                lst_unique.append(sub)
            else:
                print(sub.shape, i)
                break
        return lst_unique
    X, y = df_train.drop(['selling_price'], axis=1), df_train['selling_price']
    # print(type(X))
    # print(X)
    # print(X.duplicated())
    data = X[X.duplicated(keep=False)]
    data_sort = data.sort_values(by=list(data.columns))
    lst_data = foo(X)
    X_2 = X.iloc[list(pd.concat(lst_data).index)]
    y = y.iloc[list(X_2.index)]
    y = y.reset_index(drop=True)
    X_2 = X_2.reset_index(drop=True)
    df_train = pd.concat([X_2, y], axis=1)
    return df_train

def fill_statistics(df):
    with open('C:\\Users\\Vitaliy\\OneDrive\\Рабочий стол\\ML_HW\\hw_01\\statistics.pkl', 'rb') as f:
        statistics = pickle.load(f)
    for col, statistica in statistics.items():
        df[col].fillna(statistica, inplace=True)
    return df


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    def transform(x):
        if isinstance(x, float):
            return x
        elif isinstance(x, str):
            try:
                x = re.search('\d+', x)[0]
            except Exception:
                print(x)
                return 0
            return x
    df_test = df.copy() # чтобы не переписывать код

    cols = ['mileage', 'engine', 'max_power']
    for col in cols:
        df_test[col] = df_test[col].apply(lambda x: transform(x))


    df_test['mileage'] = df_test['mileage'].astype(float)
    df_test['engine'] = df_test['engine'].astype(float)
    df_test['max_power'] = df_test['max_power'].astype(float)

    def torque_transform(x, min=True):
        global cnt
        if x is np.nan:
            return x

        lst = re.findall(r"\d+[.,]*\d+", x)
        lst_s = re.findall(r"[a-zA-Z]+", x)

        for i in range(len(lst)):
            lst[i] = lst[i].replace(',', '.')

        if len(lst) > 2:
            mn = lst[0]
            mx = lst[-1]
        elif len(lst) == 2:
            mn = lst[0]
            mx = lst[1]
        else:
            mn = lst[0]
            mx = np.nan
            if not min:
                return mx

        if 'nm' in " ".join(lst_s).lower() and 'kgm' in " ".join(lst_s).lower():
            if lst_s[0].lower() == 'kgm':
                mn = 9.8*float(mn)

        #print(mn, x)
        if len(lst_s) >= 2:
            if lst_s[0].lower() == 'kgm':
                if min:
                    return 9.8*float(mn)
                return mx
            else:
                if min:
                    return mn
                return mx
        elif len(lst_s) == 1:
            if lst_s[0].lower() == 'rpm' or lst_s[0].lower() == 'nm':
                if min:
                    return mn
                return mx
            elif lst_s[0].lower() == 'kgm':
                if min:
                    return 9.8*float(mn)
                return mx
            else:
                print("не может быть")
        else:
            if min:
                return mn
            return mx

    df_test['max_torque_rpm'] = df_test['torque'].apply(lambda x: torque_transform(x, min=False))
    df_test['torque'] = df_test['torque'].apply(lambda x: torque_transform(x, min=True))

    df_test['torque'] = df_test['torque'].astype(float)
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].astype(float)
    return df_test


def cat_encode(df: pd.DataFrame) -> pd.DataFrame:
    df_test = df.copy()
    df_test['name'] = df_test.name.apply(lambda x: x.split()[0])

    with open('C:\\Users\\Vitaliy\\OneDrive\\Рабочий стол\\ML_HW\\hw_01\\target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)

    tst = target_encoder.transform(df_test[['name']])
    df_test['name'] = tst

    df_test['seats'] = df_test['seats'].apply(lambda x: str(x))
    cols = ['fuel','seller_type','transmission','owner', 'seats']
    for col in cols:
        with open(f'C:\\Users\\Vitaliy\\OneDrive\\Рабочий стол\\ML_HW\\hw_01\\OHE_{col}.pkl', 'rb') as f:
            ohe = pickle.load(f)
        tst = ohe.transform(df_test[[col]])[:, :-1]
        columns = [f"{col}_{i}" for i in range(tst.shape[1])]
        df_test = pd.concat([df_test.drop(col, axis=1).reset_index(drop=True), pd.DataFrame(tst, columns=columns).reset_index(drop=True)], axis=1)  

    return df_test

def generate_features(df):
    X = df.copy()
    eps = 10**(-7)
    X['feature_1'] = X['km_driven'] / (X['year'] +eps)
    X['feature_2'] = X['year'] / (X['mileage'] + eps)
    X['feature_3'] = X['km_driven'] * X['mileage']
    X['feature_4'] = X['max_power'] * X['torque']
    X['feature_5'] = X['max_torque_rpm'] / (X['max_power'] + eps)
    X['feature_6'] = 1 / X['year']
    return X

def predict(df):
    with open(f"C:\\Users\\Vitaliy\\OneDrive\\Рабочий стол\\ML_HW\\hw_01\\model.pkl", 'rb') as f:
        model = pickle.load(f)
    df = df.drop(['selling_price'], axis=1)
    pred = model.predict(df)
    return pred


@app.post("/upload_csv/")
async def predict_items_csv(file: UploadFile = File(...)) -> StreamingResponse:
    content = await file.read()
    string_data = content.decode("utf-8")
    df = pd.read_csv(StringIO(string_data))

    #df = del_duplicates(df)
    data = df.copy()

    data = transform_features(data)
    data = fill_statistics(data)
    data = cat_encode(data)
    data = generate_features(data)
    pred = predict(data)

    df['predict'] = pred

    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0) 

    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=result.csv"})
    #return pred

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    data = get_data_lst(items)
    #data = del_duplicates(data)
    data = transform_features(data)
    data = fill_statistics(data)
    data = cat_encode(data)
    data = generate_features(data)
    pred = predict(data)
    return pred

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = get_data(item)
    #data = del_duplicates(data)
    data = transform_features(data)
    data = fill_statistics(data)
    data = cat_encode(data)
    data = generate_features(data)
    pred = predict(data)
    return pred

@app.get("/")
def root():
    return {"message": "Hello World"}



    

