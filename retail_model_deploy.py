import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
lasso_model = Lasso()
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv("C:/Users/DIVYA/Desktop/PROJECT IIRS ISRO/RETAIL/retail1.csv")  # Replace 'your_dataset.csv' with the actual dataset path

# Drop the 'Crop_Year' column
data = data.drop(columns=['Customer_ID', 'Color'])

# Define features and target variable
X = data.drop(columns=['SALES'])
y = data['SALES']

# Define categorical and numerical features
categorical_features = ['Gender', 'Item_Purchased','Category', 'Location', 'Size','Season', 'Subscription_Status', 'Payment_Method', 'Shipping_Type', 'Discount_Applied', 'Promo_Code_Used', 'Preferred_Payment_Method', 'Frequency_of_Purchases']
numerical_features = ['Purchase_Amount','Review_Rating','Previous_Purchases']

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LASSO model

lasso_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', Lasso(alpha=1.0, random_state=42))])
# Train the LASSO model
lasso_model.fit(X_train, y_train)

# Predictions
y_pred = lasso_model.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

# Print the R2 score for lasso model
st.write(f'R2 Score for lasso: {r2}')


st.title('SALES RETAIL PRICE PREDICTION')

# Input fields for features
Age = st.number_input('Age', value=0.0)
Gender = st.selectbox('Select Gender', data['Gender'].unique())
Item_Purchased = st.selectbox('Select Item_Purchased', data['Item_Purchased'].unique())
Category = st.selectbox('Select Category', data['Category'].unique())
Purchase_Amount = st.number_input('Purchase_Amount', value=0.0)
Location = st.selectbox('Select Location', data['Location'].unique())
Size = st.selectbox('Select Size', data['Size'].unique())
Season = st.selectbox('Select Season', data['Season'].unique())
Review_Rating = st.number_input('Review_Rating', value=0.0)
Subscription_Status = st.selectbox(' Select Subscription_Status', data['Subscription_Status'].unique())
Payment_Method = st.selectbox('Select Payment_Method', data['Payment_Method'].unique())
Shipping_Type = st.selectbox('Select Shipping_Type', data['Shipping_Type'].unique())
Discount_Applied = st.selectbox('Select Discount_Applied', data['Discount_Applied'].unique())
Promo_Code_Used = st.selectbox('Select Promo_Code_Used', data['Promo_Code_Used'].unique())
Previous_Purchases = st.number_input('Previous Purchases ', value=0.0)
Preferred_Payment_Method = st.selectbox('Select Preferred_Payment_Method', data['Preferred_Payment_Method'].unique())
Frequency_of_Purchases = st.selectbox('Select Frequency_of_Purchases', data['Frequency_of_Purchases'].unique())

# Prepare input features
input_features = pd.DataFrame({
    'Age': [Age],
    'Gender': [Gender],
    'Item_Purchased': [Item_Purchased],
    'Category': [Category],
    'Purchase_Amount': [Purchase_Amount],
    'Location': [Location],
    'Size': [Size],
    'Season': [Season],
    'Review_Rating': [Review_Rating],
    'Subscription_Status': [Subscription_Status],
    'Payment_Method': [Payment_Method],
    'Shipping_Type': [Shipping_Type],
    'Discount_Applied': [Discount_Applied],
    'Promo_Code_Used': [Promo_Code_Used],
    'Previous_Purchases': [Previous_Purchases],
    'Preferred_Payment_Method': [Preferred_Payment_Method],
    'Frequency_of_Purchases': [Frequency_of_Purchases],
})

# Predict button
if st.button('Predict'):
    # Predict using the trained model
    prediction = lasso_model.predict(input_features)#
    st.success(f'Predicted Sales Price: {prediction[0]}')