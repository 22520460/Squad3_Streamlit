import streamlit as st
import pandas as pd
import array as arr
import numpy as np
import xgboost as xgb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree, linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title(':red[SQUAD_3] :coffee:')
st.markdown(":one: **_:blue[Upload and show Dataframe]_** :waxing_crescent_moon:")
upload_file = st.file_uploader("Choose a CSV file")

if (upload_file is not None):
    df = pd.read_csv(upload_file)  
    st.dataframe(df)
    
    #Xử lý số liệu
    df.dropna(inplace = True) # Xóa các dòng Nan
    le = LabelEncoder() # Sửa chữ thành số
    is_Category = df.dtypes == object # Gán các cột có giá trị object thì trả về True
    category_column_list = df.columns[is_Category].tolist() # List các cột có giá trị là object
    df[category_column_list] = df[category_column_list].apply(lambda col: le.fit_transform(col)) # biến object thành số
    
    # Chọn các cột để train
    st.markdown(":two: **_:blue[Choose Input Feature]_**")
    st.write("What columns do you want to use for training", str(df.columns[-1]), ':first_quarter_moon:')
    choice = arr.array('i', [])
    for i in range(0, len(df.columns) - 1):
        choice.append(1)
        choice[i] = st.checkbox(df.columns[i])
    df1 = df.copy()  # Tạo dataframe khác 
    count = False
    for i in range(0, len(df.columns) - 1):
        if (choice[i] == 0):
            del df1[df.columns[i]]
        else:
            count = True
    
    # Chọn thuật toán
    if (count): 
        st.markdown(":three: **_:blue[Choose Algorithm]_**")
        algorithm = st.selectbox(
            "Choose one of three algorithms for training :waxing_gibbous_moon::",
            ('None', 'Linear Regression', 'Decision Tree', 'XGBoost')
            ) 
        if (algorithm == 'None'):
            code = '''def buffet() :
            free = 'please choose again'
            print(free)'''
            st.code(code, 'python')        
        else:
            st.markdown(":four: **_:blue[Drawing explicity chart]_**")
            
            # Chọn tỉ lệ train/test
            ratio = st.slider('Choose ratio train spilt :full_moon::', 0.0, 1.0, 0.25)
            if (ratio == 1 or ratio == 0):
                code = '''def buffet() :
                free = 'please choose again'
                print(free)'''
                st.code(code, 'python')  
            else:
                train, test = train_test_split(df1, train_size = ratio, random_state = 40)

                x_train = train.drop(columns = [df1.columns[-1]])
                y_train = train[df1.columns[-1]]
                
                x_test = test.drop(columns = [df1.columns[-1]])
                y_test = test[df1.columns[-1]]
                
                # Tính toán số liệu
                y_pred = []
                if (algorithm == 'XGBoost') :
                    model_xgb = xgb.XGBRegressor(random_state=50,learning_rate = 0.2, n_estimators = 100)
                    model_xgb.fit(x_train, y_train)
                    y_pred = model_xgb.predict(x_test)

                if (algorithm == 'Decision Tree') :
                    model_dcs_tree = tree.DecisionTreeRegressor(min_samples_leaf = 4, min_samples_split = 4, random_state=0)
                    model_dcs_tree.fit(x_train, y_train)
                    y_pred = model_dcs_tree.predict(x_test)

                if (algorithm == 'Linear Regression') :
                    model_regr = linear_model.LinearRegression(random_state=50,learning_rate = 0.1)
                    model_regr.fit(x_train, y_train)
                    y_pred = model_regr.predict(x_test)

                MAE_1 = mean_absolute_error(y_test, y_pred)
                MAE_2 = mean_absolute_error(y_pred, y_test)
                MSE_1 = mean_squared_error(y_test, y_pred, squared=False)
                MSE_2 = mean_squared_error(y_pred, y_test, squared=False)

                #Vẽ đồ thị 
                x = ['MAE_1', 'MAE_2', 'MSE_1', 'MSE_2']
                y = [MAE_1, MAE_2, MSE_1, MSE_2]

                fig, ax = plt.subplots()
                ax = plt.gca()
                ax.set_ylim(-100, 30000)
                plt.bar(x, y, color = ('lightsalmon', 'lightgreen', 'lightsalmon', 'lightgreen')) # Lấy màu cho các cột        
                plt.title(algorithm + "(use logarithm)", fontsize = 14)
                
                # Ghi chú thích
                lightsalmon = mpatches.Patch(color = 'lightsalmon', label = 'y_test/y_train')
                lightgreen = mpatches.Patch(color = 'lightgreen', label = 'y_train/y_test')
                plt.legend(handles = [lightsalmon, lightgreen])
                plt.ylabel("log", fontsize = 12)
                
                # In ra giá trị
                for i in range (0, 4):
                    plt.text(x[i], y[i] + 0.001, str(round(y[i], 4)), transform = plt.gca().transData, horizontalalignment = 'center', color = 'black', fontsize = 'medium')
                st.pyplot(fig)
