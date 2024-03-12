import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import sklearn
import pickle
import matplotlib.pyplot as plt

st.markdown("<div style='text-align: center;'><h1 style='color: blue;'>ECOMMERCE APP & WEBSITE USAGE PREDICTION</h1></div>", unsafe_allow_html=True)
# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# Check if a file is uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    #data = pickle.load(open('ecommerce.pkl','rb'))
	data = pd.read_csv('Ecommerce_Customers.csv')
    # Display the DataFrame
else:
    st.info('Ecommerce_Customers.csv')

st.sidebar.header("CONTENTS")
# Sidebar with buttons
selected_option = st.sidebar.radio("Select the Content", ["Introduction", "EDA", "Visualization","Model Building"])
# Display the DataFrame
if selected_option == "Introduction":
# Streamlit app
	st.write("## Problem Statement")
	st.write("""A project with an Ecommerce company sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want. The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They have asked to help them figure it out.

	• Avg. Session Length: Average session of in-store style advice sessions in minutes.

	• Time on App: Average time spent on App in minutes

	• Time on Website: Average time spent on Website in minutes

	• Length of Membership: How many years the customer has been a member.

	• Yearly Amount Spent: The total amount the customer is spending in dollars.""")

# Display the loaded data
	st.write("### Data from CSV File")
	st.write(data)

df1 = data
df2 = df1.rename({'Avg Session Length':'ASL', 'Time on App' : 'TOA', 'Time on Website' :'TOW', 'Length of Membership':'LOM', 'Yearly Amount Spent' : 'YAS'}, axis=1)
df3 = df2.drop(columns=df1[['Email','Address','Avatar']],axis =1)
#Outliers
Outliers = df2[(df2["ASL"] > 34.397067) | (df2["ASL"] < 31.656740)&
               (df2["TOA"] >  13.436698) | (df2["TOW"] < 10.705305 )&
               (df2["TOW"] > 38.400020) | (df2["TOW"] < 35.665670 )&
               (df2["LOM"] >  4.724528) | (df2["LOM"] < 2.332424)&
               (df2["YAS"] > 601.451603) | (df2["YAS"] < 392.900502 )]

med=df3.median()
#Outliers are imputed with median
# Create a DataFrame from med to facilitate broadcasting
med_df = pd.DataFrame(np.tile(med, (Outliers.shape[0], 1)), columns=df3.columns)
# Ensure outliers array has a boolean data type
outliers_bool = Outliers.astype(bool)
# Replace outliers in df1 with median values using boolean indexing
df4= df3.copy()
df4[outliers_bool] = med_df[outliers_bool]
df4 = df4.fillna(df4.median())
# EDA
if selected_option == "EDA" :
	st.title("EDA")
	st.write("### The head data",df1.head())
	st.write("### The Shape of the data",df1.shape)
	st.write("### The Datatypes of dataset",df1.dtypes)
	st.write("### Descriptive Statistics of the data",df1.describe())
	st.write("### Missing values in dataset",df1.isnull().sum())
	st.write("### Renaming the columns",df2.columns)
	st.write("### Checking Duplicate",df2[df2.duplicated()].sum())
	st.write("### Dropped unnecessary columns",df3.head())
	st.write("### Outliers",Outliers)
	st.write("### Imputed data which is treated with median",df4)

#Visualization
if selected_option == "Visualization" :
#Line plots
	plt.figure(figsize=(10,6))
	plt.suptitle('Comparision of Original data & Imputed data using line plot',fontsize=20,color='red')
	plt.subplot(5,2,1)
	sns.lineplot(df3.iloc[:,:-2])
	plt.title("Original data")
	plt.subplot(5,2,2)
	sns.lineplot(df4.iloc[:,:-2])
	plt.title('Imputed data')
	plt.ylabel('minutes')
	plt.show()
	st.pyplot(plt)
	plt.figure(figsize=(10,4))
	plt.subplot(1,2,1)
	sns.lineplot(df3['LOM'],label='Original LOM')
	sns.lineplot(df4['LOM'],label='Imputed LOM')
	plt.title('Lenght of Membership',color='red')
	plt.subplot(1,2,2)
	sns.lineplot(df3['YAS'],label='Original YAS')
	sns.lineplot(df4['YAS'],label='Imputed YAS')
	plt.title('Yearly amount spent',color='red')
	plt.show()
	st.pyplot(plt)
#Boxplots
	plt.figure(figsize=(7,5))
	plt.suptitle("Boxplots",fontsize=20,color='red')
	plt.subplot(1,2,1)
	sns.boxplot(df3.iloc[:,:-2])
	plt.title('Original Data')
	plt.subplot(1,2,2)
	sns.boxplot(df4.iloc[:,:-2])
	plt.title('Imputed Data')
	plt.show()
	st.pyplot(plt)
	plt.figure(figsize=(7,5))
	plt.suptitle('Lenght of Membership',color='red')
	plt.subplot(1,2,1)
	sns.boxplot(df3['LOM'])
	plt.title('Original data')
	plt.subplot(1,2,2)
	sns.boxplot(df4['LOM'])
	plt.title('Imputed data')
	plt.show()
	st.pyplot(plt)
	plt.figure(figsize=(10,5))
	plt.suptitle('Yearly Amount Spent',color='red')
	plt.subplot(1,2,1)
	sns.boxplot(df3['YAS'])
	plt.title('Original data')
	plt.subplot(1,2,2)
	sns.boxplot(df4['YAS'])
	plt.title('Imputed data')
	plt.show()
	st.pyplot(plt)
#Regressionplot
	plt.figure(figsize=(22,8))
	plt.suptitle("Regressionplot for Original data",color='red',fontsize=30)
	plt.subplot(1,4,1)
	sns.regplot(x='TOA',y='YAS',data=df3)
	plt.subplot(1,4,2)
	sns.regplot(x='TOW',y='YAS',data=df3)
	plt.subplot(1,4,3)
	sns.regplot(x='LOM',y='YAS',data=df3)
	plt.subplot(1,4,4)
	sns.regplot(x='ASL',y='YAS',data=df3)
	plt.show()
	st.pyplot(plt)
	plt.figure(figsize=(22,8))
	plt.suptitle("Regressionplot for Imputed data",color='red',fontsize=30)
	plt.subplot(1,4,1)
	sns.regplot(x='TOA',y='YAS',data=df4)
	plt.subplot(1,4,2)
	sns.regplot(x='TOW',y='YAS',data=df4)
	plt.subplot(1,4,3)
	sns.regplot(x='LOM',y='YAS',data=df4)
	plt.subplot(1,4,4)
	sns.regplot(x='ASL',y='YAS',data=df4)
	plt.show()
	st.pyplot(plt)
#p-p plot
	plt.figure(figsize=(22,8))
	plt.suptitle('P-P Plots for Original data',fontsize=40,color='red')
	plt.subplot(1,5,1)
	stats.probplot(df3['ASL'], dist="norm", plot=plt)
	plt.subplot(1,5,2)
	stats.probplot(df3['TOA'], dist="norm", plot=plt)
	plt.subplot(1,5,3)
	stats.probplot(df3['TOW'], dist="norm", plot=plt)
	plt.subplot(1,5,4)
	stats.probplot(df3['LOM'], dist="norm", plot=plt)
	plt.subplot(1,5,5)
	stats.probplot(df3['ASL'], dist="norm", plot=plt)
	plt.legend()
	st.pyplot(plt)
	plt.figure(figsize=(22,8))
	plt.suptitle('P-P Plots for Imputed data',fontsize=40,color='red')
	plt.subplot(1,5,1)
	stats.probplot(df4['ASL'], dist="norm", plot=plt)
	plt.subplot(1,5,2)
	stats.probplot(df4['TOA'], dist="norm", plot=plt)
	plt.subplot(1,5,3)
	stats.probplot(df4['TOW'], dist="norm", plot=plt)
	plt.subplot(1,5,4)
	stats.probplot(df4['LOM'], dist="norm", plot=plt)
	plt.subplot(1,5,5)
	stats.probplot(df4['ASL'], dist="norm", plot=plt)
	plt.legend()
	st.pyplot(plt)
#pairplot
	st.write('### Pairplot for Original data')
	sns.pairplot(df3)
	plt.show()
	st.pyplot(plt)
	st.write('### Pairplot for Imputed data')
	sns.pairplot(df4)
	plt.show()
	st.pyplot(plt)
	st.write("### Correlation",df3.corr())
#Correlation Heatmap
	correlation_matrix1 = df3.corr()
	plt.figure(figsize=(22,8))
	plt.subplot(1,2,1)
	sns.heatmap(correlation_matrix1, annot=True, cmap="coolwarm")
	plt.title("Original data Correlation",fontsize=24)
	plt.subplot(1,2,2)
	correlation_matrix2 = df4.corr()
	sns.heatmap(correlation_matrix2, annot=True, cmap="coolwarm")
	plt.title("Imputed data Correlation",fontsize=24)
	plt.show()
	st.pyplot(plt)
#Scatterplot
	plt.figure(figsize=(22,8))
	plt.suptitle('Scatter Plots',fontsize=40,color='red')
	# Fit a linear regression line
	slope, intercept = np.polyfit(df3['TOA'], df3['YAS'],1)
	trend_line = slope * df3['TOA'] + intercept
	# Create a scatter plot
	plt.scatter(df3['TOA'], df3['YAS'],label='Original data')
	plt.scatter(df4['TOA'],df4['YAS'],label='Imputed data')
	# Plot the trend line
	plt.plot(df3['TOA'], trend_line, color='red', label='Trend Line')
	plt.xlabel('Time on App (minutes)')
	plt.ylabel('Yearly Amount Spent ($)')
	plt.title('Time on App vs Yearly Amount Spent')
	plt.legend()
	st.pyplot(plt)
	plt.figure(figsize=(22,8))
	# Fit a linear regression line
	slope, intercept = np.polyfit(df3['TOW'], df3['YAS'],1)
	trend_line = slope * df3['TOW'] + intercept
	# Create a scatter plot
	plt.scatter(df3['TOW'], df3['YAS'],label='Original data')
	plt.scatter(df4['TOW'],df4['YAS'],label='Imputed data')
	# Plot the trend line
	plt.plot(df3['TOW'], trend_line, color='red', label='Trend Line')
	plt.xlabel('Time on website (minutes)')
	plt.ylabel('Yearly Amount Spent ($)')
	plt.title('Time on website vs Yearly Amount Spent')
	plt.legend()
	st.pyplot(plt)
	plt.figure(figsize=(22,8))
	# Fit a linear regression line
	slope, intercept = np.polyfit(df3['LOM'], df3['YAS'],1)
	trend_line = slope * df3['LOM'] + intercept
	# Create a scatter plot
	plt.scatter(df3['LOM'], df3['YAS'],label='Original data')
	plt.scatter(df4['LOM'],df4['YAS'],label='Imputed data')
	# Plot the trend line
	plt.plot(df3['LOM'], trend_line, color='red', label='Trend Line')
	plt.xlabel('lenght of membership (YEARS)')
	plt.ylabel('Yearly Amount Spent ($)')
	plt.title('lenght of membership vs Yearly Amount Spent')
	plt.legend()
	st.pyplot(plt)
	plt.figure(figsize=(22,8))
	# Fit a linear regression line
	slope, intercept = np.polyfit(df3['ASL'], df3['YAS'],1)
	trend_line = slope * df3['ASL'] + intercept
	# Create a scatter plot
	plt.scatter(df3['ASL'], df3['YAS'],label='Original data')
	plt.scatter(df4['ASL'],df4['YAS'],label='Imputed data')
	# Plot the trend line
	plt.plot(df3['ASL'], trend_line, color='red', label='Trend Line')
	plt.xlabel('Average session lenght (minutes)')
	plt.ylabel('Yearly Amount Spent ($)')
	plt.title('Average session lenght vs Yearly Amount Spent')
	plt.legend()
	st.pyplot(plt)
#Density plot
	st.write("## Density plot")
	plt.figure(figsize=(22,8))
	plt.subplot(1,2,1)
	sns.histplot(df3['TOA'], bins=20, kde=True, label='original data',color='pink')
	sns.histplot(df4['TOA'], bins=20, kde=True, label='imputed data')
	plt.xlabel('Time on App')
	plt.ylabel('Count')
	plt.legend()
	plt.subplot(1,2,2)
	sns.histplot(df3['TOW'], bins=20, kde=True, label='original data')
	sns.histplot(df4['TOW'], bins=20, kde=True, label='imputed data')
	plt.xlabel('Time on Website')
	plt.ylabel('Count')
	plt.legend()
	plt.show()
	st.pyplot(plt)
#Skewness
	st.write("### Skewplots for Original data", df3.skew())
	st.write("### Skewplots for Imputed data", df4.skew())
# Create a histogram with kernel density estimate
	plt.figure(figsize=(22,8))
	plt.suptitle('Skewness plots',fontsize=40,color='red')
	plt.subplot(1,2,1)
	sns.histplot(df3[['ASL','TOA','TOW','LOM']],kde=True)
	plt.title('Original data Skewness Plot')
	plt.subplot(1,2,2)
	sns.histplot(df4[['ASL','TOA','TOW','LOM']],kde=True)
	plt.title('Imputed data Skewness Plot')
	st.pyplot(plt)
	st.write("""**Original Dataset (df3):**

Avg Session Length (ASL): The skewness is close to zero (-0.032). This suggests that the distribution is approximately symmetrical.

Time on App (TOA): The skewness is negative (-0.089). This indicates a slight left-skewness, suggesting that the distribution has a longer left tail.

Time on Website (TOW): The skewness is close to zero (0.012). This suggests that the distribution is approximately symmetrical.

Length of Membership (LOM): The skewness is negative (-0.107). This indicates a slight left-skewness, suggesting that the distribution has a longer left tail.

Yearly Amount Spent (YAS): The skewness is close to zero (0.035). This suggests that the distribution is approximately symmetrical.""")
	st.write("""**Imputed Dataset (df4):**

Avg Session Length (ASL): The skewness is more negative (-0.587) compared to the original dataset. This suggests an increased left-skewness, possibly indicating an elongated left tail in the distribution after imputation.

Time on App (TOA):The skewness is slightly positive (0.064) after imputation. This suggests a right-skewness, possibly indicating an elongated right tail in the distribution after imputation.

Time on Website (TOW): The skewness is positive (0.111) after imputation. This indicates right-skewness, suggesting an elongated right tail in the distribution after imputation.

Length of Membership (LOM): The skewness is more positive (0.208) compared to the original dataset. This suggests increased right-skewness, possibly indicating an elongated right tail in the distribution after imputation.

Yearly Amount Spent (YAS): The skewness is more positive (0.775) compared to the original dataset. This indicates increased right-skewness, possibly indicating an elongated right tail in the distribution after imputation.""")
X = df3.iloc[:,:-1]
Y= df3.iloc[:,-1]
	
#Model Building
if selected_option == "Model Building" :
	st.write("### The Scaled training and Testing data of 'X' variable")
	data_percentage = st.slider("Select Percentage of  Data", 1, 100, 80)
	# Using st.columns
	col1, col2 = st.columns(2)
	from sklearn.model_selection import train_test_split
	train_size = int((data_percentage / 100) * len(X))
	X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=train_size, random_state=0)
	from sklearn.preprocessing import MinMaxScaler
	# Initialize the MinMaxScaler
	scaler = MinMaxScaler()
	# Fit and transform the X_train and X_test data
	X_train = scaler.fit_transform(X_train)
	X_test= scaler.transform(X_test)
	# Fit and transform the y_train and y_test data
	y_train = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
	y_test = scaler.transform(y_test.values.reshape(-1, 1)).flatten()
	# Display DataFrame 1 in the first column
	st.write(f"Selected Percentage: {data_percentage}%")
	col1.header("Training data")
	col1.write(f"Number of Training Samples: {len(X_train)}")
	col1.write(X_train)
	# Display DataFrame 2 in the second column
	col2.header("Testing data")
	col2.write(f"Number of Testing Samples: {len(X_test)}")
	col2.write(X_test)
	st.write("### The Scaled training and Testing data of 'Y' variable")
	col1, col2 = st.columns(2)
	col1.header("Training data")
	col1.write(f"Number of Training Samples: {len(y_train)}")
	col1.write(y_train)
	# Display DataFrame 2 in the second column
	col2.header("Testing data")
	col2.write(f"Number of Testing Samples: {len(y_test)}")
	col2.write(y_test)
	#Linear regression
	from sklearn.linear_model import LinearRegression
	from sklearn.metrics import r2_score, mean_squared_error
	from math import sqrt
	# Initialize the Linear Regression model
	modell = LinearRegression()
	# Fit the model on the training data
	modell.fit(X_train, y_train)
	# Predict the target variable on the test set
	y_pred = modell.predict(X_test)
	# Calculate R-squared score
	r2 = r2_score(y_test, y_pred)
	# Calculate RMSE
	rmse = sqrt(mean_squared_error(y_test, y_pred))
	# Print the results
	st.write("Linear Regression model R-squared Score:", r2)
	st.write("Linear Regression model Root Mean Squared Error (RMSE):", rmse)
	from sklearn.tree import DecisionTreeRegressor
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.svm import SVR
	from sklearn.neighbors import KNeighborsRegressor
	from sklearn.linear_model import Ridge
	from sklearn.linear_model import Lasso
	from sklearn.linear_model import ElasticNet
	# Define a list of models
	models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    SVR(),
    KNeighborsRegressor(),
    ElasticNet(),
    Ridge(),
    Lasso()
# Add more models as needed
	]
# Create a list to store results
	results_list = []
# Loop through each model
	for model in models:
    # Fit the model
	    model.fit(X_train, y_train)

    # Make predictions
	    y_pred = model.predict(X_test)

    # Calculate R2 score and RMSE
	    r2 = r2_score(y_test, y_pred)
	    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Append results to the list
	    results_list.append({'Model': type(model).__name__, 'R2 Score': r2, 'RMSE': rmse})

# Create a DataFrame from the list of results
	results_df = pd.DataFrame(results_list)

# Display the results DataFrame
	st.write(results_df)
	# Coefficients of the LinearRegression model
	coefficients = modell.coef_
	intercept = modell.intercept_
	st.write("Linear regression model Coefficients:", coefficients)
	st.write("Linear regression model 7410Intercept:", intercept)
	plt.figure(figsize=(8, 6))
	plt.bar(X.columns, coefficients)
	plt.title('Linear Regression Coefficients',color='red')
	plt.xlabel('Features')
	plt.ylabel('Coefficient Values')
	plt.show()
	st.pyplot(plt)
	# Assuming coefficients are in the same order as ['ASL', 'TOA', 'TOW', 'LOM']
	feature_names = ['ASL', 'TOA', 'TOW', 'LOM']
	for feature, coef in zip(feature_names, coefficients):
		st.write(f"{feature}: {coef}")
	# Identify the feature with the highest positive coefficient
	max_positive_coef_feature = feature_names[np.argmax(coefficients)]