import streamlit as st
import pandas as pd
import array as arr
import numpy as np
import xgboost as xgb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import tree, linear_model
from io import BytesIO

st.title('SQUAD 3')
st.markdown("**:red[Upload and show Dataframe]**")
upload_file = st.file_uploader("Choose a CSV file")
if upload_file is not None:
    df = pd.read_csv(upload_file)  

    #Xử lý số liệu
    df.dropna(inplace=True) # Xóa các dòng Nan
    le = LabelEncoder() # Sửa chữ thành số
    is_Category = df.dtypes == object # Gán các cột có giá trị object thì trả về True
    category_column_list = df.columns[is_Category].tolist() # List các cột có giá trị là object
    df[category_column_list] = df[category_column_list].apply(lambda col: le.fit_transform(col)) # biến object thành số

    st.dataframe(df)
    
    # Chọn các cột để train
    st.markdown("**:red[Choose Input Feature]**")
    st.write("What columns do you want to use for training " ,str(df.columns[-1]))
    choice1 = arr.array('i',[])
    for i in range(0, len(df.columns) - 1):
        choice1.append(i)
        choice1[i]= st.checkbox(df.columns[i])
        
    df1 = df # Tạo dataframe khác 
    dem = 0 # Đếm số cột chọn cho train
    for i in range(0, len(df.columns) - 2):
        if (choice1[i] == False):
            del df1[df.columns[i]]
        else:
            dem +=1
    
    # Chọn thuật toán
    if (dem >= 1): 
        st.markdown("**:red[Choose Algorithm]**")
        algorithm = st.selectbox(
        "Choose one of three algorithms for training :",
        ('Linear Regression', 'Decision Tree', 'XGBoost')
        )

        st.markdown("**:red[Drawing explicity chart]**")
        # Chọn tỉ lệ train/test
        ratio = st.slider('Choose ratio train/test spilt :', 0.0, 1.0, 0.25)
        train, test = train_test_split(df1, train_size = ratio, random_state = 40)

        x_train = train.drop(columns = [df1.columns[-1]])
        y_train = train[df1.columns[-1]]

        x_test = test.drop(columns = [df1.columns[-1]])
        y_test = test[df1.columns[-1]]
        # Tính toán số liệu
        MAE_1 = 0
        MAE_2 = 0
        MSE_1 = 0
        MSE_2 = 0
        if (algorithm == 'XGBoost') :
            model_xgb = xgb.XGBRegressor(random_state=50,learning_rate = 0.2, n_estimators = 100)
            model_xgb.fit(x_train, y_train)
            y_pred_XGB = model_xgb.predict(x_test)
            MAE_1 = mean_absolute_error(y_test, y_pred_XGB)
            MAE_2 = mean_absolute_error(y_pred_XGB, y_test)
            MSE_1 = mean_squared_error(y_test, y_pred_XGB, squared=False)
            MSE_2 = mean_squared_error(y_pred_XGB, y_test, squared=False)
            
        if (algorithm == 'Decision Tree') :
            model_dcs_tree = tree.DecisionTreeRegressor(min_samples_leaf = 4, min_samples_split = 4, random_state=0)
            model_dcs_tree.fit(x_train, y_train)
            y_pred_dcs_tree = model_dcs_tree.predict(x_test)
            MAE_1 = mean_absolute_error(y_test, y_pred_dcs_tree)
            MAE_2 = mean_absolute_error(y_pred_dcs_tree, y_test)
            MSE_1 = mean_squared_error(y_test, y_pred_dcs_tree, squared=False)
            MSE_2 = mean_squared_error(y_pred_dcs_tree, y_test, squared=False)

        if (algorithm == 'Linear Regression') :
            model_regr = linear_model.LinearRegression()
            model_regr.fit(x_train, y_train)
            y_pred_regr = model_regr.predict(x_test)
            MAE_1 = mean_absolute_error(y_test, y_pred_regr)
            MAE_2 = mean_absolute_error(y_pred_regr, y_test)
            MSE_1 = mean_squared_error(y_test, y_pred_regr, squared=False)
            MSE_2 = mean_squared_error(y_pred_regr, y_test, squared=False)
            
        #Vẽ đồ thị 
        x = ['MAE_1','MAE_2','MSE_1','MSE_2']
        y = [MAE_1,MAE_2,MSE_1,MSE_2]

        fig, ax = plt.subplots()
        plt.bar(x,y,color=('lightsalmon','lightgreen', 'lightsalmon', 'lightgreen')) # Lấy màu cho các cột
            # Ghi chú thích
        lightsalmon = mpatches.Patch(color='lightsalmon', label='y_test/y_train')
        lightgreen = mpatches.Patch(color='lightgreen', label='y_train/y_test')
        plt.legend(handles=[lightsalmon,lightgreen])
            # In ra giá trị
        for i in range (0,4):
            plt.text(x[i],y[i] + 100,str(round(y[i],2)), transform = plt.gca().transData,horizontalalignment = 'center', color = 'black',fontsize = 'medium')
            # Ép biểu đồ về dạng ảnh rồi thu nhỏ lại
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf, width = 600)
