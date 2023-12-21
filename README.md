# MLTT-Assignment 1
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# Load a smaller subset of the California Housing dataset for faster testing
data = fetch_california_housing()
X, _, y, _ = train_test_split(data.data, data.target, test_size=0.9, random_state=42)

# Split the smaller subset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with feature selection and regressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_regression, k='all')),  # Set k to 'all' for all features
    ('regressor', RandomForestRegressor())
])

# Define hyperparameters for RandomizedSearchCV
param_dist = {
    'feature_selection__k': [5, 8, 'all'],  # Adjust the values based on the number of features in your dataset
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__bootstrap': [True, False]
}

# Use RandomizedSearchCV for hyperparameter tuning with fewer iterations
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=5, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1, error_score='raise')

# Fit the model
random_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters: ", random_search.best_params_)

# Evaluate the model on the test set
y_pred = random_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Best Mean Squared Error from Cross-Validation: ", -random_search.best_score_)  # Best MSE from cross-validation
print("Root Mean Squared Error on Test Set: ",rmse)







# MLTT Assignment 2
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "71f49196-cce9-4afe-8923-1a123e38adf0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "# Data Visualization\n",
    "import seaborn as sn\n",
    "import matplotlib.pyplot as plt\n",
    "# K-Means Cluster\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.preprocessing import OrdinalEncoder\n",
    "encoder = OrdinalEncoder()\n",
    "from sklearn.cluster import KMeans\n",
    "df = pd.read_csv(r'C:\\Users\\Chaithra.k\\OneDrive\\Desktop\\telecom_churn.csv')\n",
    "# Inspect Data \n",
    "df.head(2)\n",
    "df[\"SeniorCitizen\"]= df[\"SeniorCitizen\"].map({0: \"No\", 1: \"Yes\"})\n",
    "# Each row represents a customer, each column contains customer’s attributes described on the column Metadata.\n",
    "df.isnull().sum(axis = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "110c5a6e-00a5-4c57-a7c9-5e58d94a7a7a",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Output Variable\n",
    "fig, ax = plt.subplots(1,1, figsize=(8, 6))\n",
    "data_temp = df['Churn'].value_counts().sort_index()\n",
    "ax.bar(data_temp.index, data_temp,\n",
    "          edgecolor='black', color='#d4dddd',\n",
    "          width=0.55 )\n",
    "ax.set_title('Churn', loc='left', fontsize=19, fontweight='bold')\n",
    "for i in data_temp.index:\n",
    "    ax.annotate(f\"{data_temp[i]}\", \n",
    "                   xy=(i, data_temp[i] + 100),\n",
    "                   va = 'center', ha='center',fontweight='light', fontfamily='serif',\n",
    "                   color='black')\n",
    "for s in ['top', 'right']:\n",
    "    ax.spines[s].set_visible(False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2e291414-c188-4477-9015-1c1d36131a22",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "def CountPlot_Table (feature):\n",
    "    # Create Count Plot for Churn Vs Feature\n",
    "    sn.countplot(x=feature, hue=\"Churn\", data=df, palette=\"Paired\", edgecolor = 'Black', order=df[feature].value_counts().index)\n",
    "    sn.despine()\n",
    "    # Create a plot for proportions\n",
    "    temp_table = pd.DataFrame(round(df.groupby(feature)['Churn'].value_counts(normalize = True),4))\n",
    "    table = plt.table(cellText=temp_table.values,\n",
    "          rowLabels=temp_table.index,\n",
    "          colLabels=temp_table.columns,\n",
    "          bbox=(1.5, 0,0.4 , 0.45))\n",
    "    table.auto_set_font_size(False)\n",
    "    table.set_fontsize(12)\n",
    "    plt.show()\n",
    "demo_features = ['gender','SeniorCitizen','Partner','Dependents']\n",
    "for feature in demo_features:\n",
    "    CountPlot_Table(feature)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e52f3d76-7770-4849-b5ea-98832ef36c65",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Churn Rate comparision amongst demographics\n",
    "df2 =df.copy()\n",
    "df2[\"Churn\"]= df2[\"Churn\"].map({\"No\" : 0 , \"Yes\": 1})\n",
    "sn.set_theme(style ='whitegrid')\n",
    "g=sn.PairGrid(df2, y_vars = 'Churn',\n",
    "             x_vars = ['gender','SeniorCitizen','Partner','Dependents'], height = 5, aspect =0.75)\n",
    "g.map(sn.pointplot, scale = 1, errwidth =2, color = 'xkcd:plum')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8e0f1e21-10a0-43a5-8f5c-88dd163933c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import scipy.stats as stats\n",
    "import matplotlib.pyplot as plt\n",
    "import math\n",
    "\n",
    "Male_Churn = df2[df2[\"gender\"] == 'Male'].Churn  \n",
    "Female_Churn = df2[df2[\"gender\"] == 'Female'].Churn\n",
    "\n",
    "t_statstics = stats.ttest_ind(a= Male_Churn,\n",
    "                b= Female_Churn,\n",
    "                equal_var=False)    # Assume samples have equal variance?\n",
    "t_statstics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8ce847a7-77e6-49ae-b83c-183aea84c14f",
   "metadata": {},
   "outputs": [],
   "source": [
    "Dependents_No = df2[df2[\"Dependents\"] == 'No'].Churn\n",
    "Dependents_Yes = df2[df2[\"Dependents\"] == 'Yes'].Churn\n",
    "\n",
    "t_statstics1 = stats.ttest_ind(a= Dependents_No, b= Dependents_Yes, equal_var=False)   \n",
    "print(t_statstics1)\n",
    "t_statstics2 = stats.ttest_ind(a= df2[df2[\"Partner\"] == 'No'].Churn, b=  df2[df2[\"Partner\"] == 'Yes'].Churn, equal_var=False)   \n",
    "print(t_statstics2)\n",
    "t_statstics3 = stats.ttest_ind(a= df2[df2[\"SeniorCitizen\"] == 'No'].Churn, b=  df2[df2[\"SeniorCitizen\"] == 'Yes'].Churn, equal_var=False)\n",
    "print(t_statstics3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b63c1928-3d36-4e0d-9b53-bf7b2fec5871",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = df['PaymentMethod'].value_counts()\n",
    "keys = df['PaymentMethod'].unique()  \n",
    "\n",
    "# declaring exploding pie\n",
    "explode = [0.1, 0, 0, 0]\n",
    "# define Seaborn color palette to use\n",
    "palette_color = sn.color_palette('Paired')\n",
    "# plotting data on chart\n",
    "\n",
    "fig, ax = plt.subplots(1, 2, figsize=(20, 7))\n",
    "ax[0].pie(data, labels=keys, colors=palette_color,\n",
    "        explode=explode, autopct='%.0f%%')\n",
    "\n",
    "# create data\n",
    "\n",
    "yes_churn = [258, 232, 1071, 308]\n",
    "no_churn = [1286, 1290,1294, 1304]\n",
    "ax[1].bar(keys, yes_churn, label='Churn', color = 'skyblue',edgecolor='white', width = 0.7)\n",
    "ax[1].bar(keys, no_churn, label='No Churn', bottom=yes_churn, color = 'forestgreen', edgecolor='white', width = 0.7)\n",
    "ax[1].legend()\n",
    "fig.text(0.60, 0.92, 'Payment Method vs Churn', fontsize=17, fontweight='bold')    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76d9eaf8-57be-4bea-8c36-573f0b2477e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "Churn_0 = df2[df2[\"Churn\"] == 0]\n",
    "Churn_1 = df2[df2[\"Churn\"] == 1]\n",
    "# plotting first histogram\n",
    "fig, ax = plt.subplots(1, 2, figsize=(18, 7))\n",
    "ax[0].hist(Churn_0.tenure, label='No Churn', alpha=.8, edgecolor='darkgrey')\n",
    "# plotting second histogram\n",
    "ax[0].hist(Churn_1.tenure, label='Churn', alpha=0.7, edgecolor='pink')\n",
    "ax[0].legend()  \n",
    "\n",
    "columns = [Churn_0.tenure, Churn_1.tenure]\n",
    "ax[1].boxplot(columns, notch=True, patch_artist=True)\n",
    "plt.xticks([1, 2], [\"No Churn\", \"Churn\"])\n",
    "fig.text(0.45, 0.92, 'Tenure vs Churn', fontsize=17, fontweight='bold') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "705a0bdc-875d-4359-9828-a35c77731090",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(9,7))\n",
    "ax = sn.countplot(x=\"Contract\", hue=\"Churn\", data=df).set(title='Contracts vs Churn Rates', xlabel=None, ylabel = None)\n",
    "sn.despine()\n",
    "plt.legend(title='', loc='upper right', labels=['No Churn', 'Churn'])\n",
    "plt.show(g)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e934354-51c8-4625-bd56-b7bc08d872d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "ax = sn.kdeplot(Churn_0.MonthlyCharges, color=\"#9C7FE8\", shade = True)\n",
    "ax = sn.kdeplot(Churn_1.MonthlyCharges, color=\"#00677C\", shade = True)\n",
    "ax.legend([\"No Churn\",\"Churn\"],loc='upper right')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c7010d77-322a-42a4-bec8-a54423aea737",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Total Charges\n",
    "pd.set_option('mode.chained_assignment', None)\n",
    "Churn_0['TotalCharges'] = pd.to_numeric(Churn_0['TotalCharges'],errors = 'coerce')\n",
    "Churn_1['TotalCharges'] = pd.to_numeric(Churn_1['TotalCharges'],errors = 'coerce')\n",
    "\n",
    "ax = sn.kdeplot(Churn_0.TotalCharges, color=\"#9C7FE8\", shade = True)\n",
    "ax = sn.kdeplot(Churn_1.TotalCharges, color=\"#00677C\", shade = True)\n",
    "ax.legend([\"No Churn\",\"Churn\"],loc='upper right')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4f5c17ce-138b-406b-8447-6ef1851718bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14,12))\n",
    "# Gray for No Churn, highlight Churn!\n",
    "colors = [\"#C7CDCB\", \"#781B24\"]\n",
    "# Set custom color palette\n",
    "sn.set_palette(sn.color_palette(colors))\n",
    "# Graphing\n",
    "sn.countplot(x=\"PhoneService\", hue=\"Churn\", data=df, ax=axes[0,0]).set(title='Phone Services vs Churn Rates', xlabel=None, ylabel = None)\n",
    "sn.countplot(x=\"MultipleLines\", hue=\"Churn\", data=df, ax=axes[0,1]).set(title='Multiple Lines Services vs Churn Rates', xlabel=None, ylabel = None)\n",
    "sn.countplot(x=\"InternetService\", hue=\"Churn\", data=df, ax=axes[0,2]).set(title='Internet Service vs Churn Rates', xlabel=None, ylabel = None)\n",
    "sn.countplot(x=\"OnlineSecurity\", hue=\"Churn\", data=df, ax=axes[1,0]).set(title='Online Security Services vs Churn Rates', xlabel=None, ylabel = None)\n",
    "sn.countplot(x=\"DeviceProtection\", hue=\"Churn\", data=df, ax=axes[1,1]).set(title='Device Protection Services vs Churn Rates', xlabel=None, ylabel = None)\n",
    "sn.countplot(x=\"TechSupport\", hue=\"Churn\", data=df, ax=axes[1,2]).set(title='Tech Support Services vs Churn Rates', xlabel=None, ylabel = None)\n",
    "sn.countplot(x=\"StreamingTV\", hue=\"Churn\", data=df, ax=axes[2,0]).set(title='Streaming Tv Services vs Churn Rates', xlabel=None, ylabel = None)\n",
    "sn.countplot(x=\"StreamingMovies\", hue=\"Churn\", data=df, ax=axes[2,1]).set(title='Streaming Movies Services vs Churn Rates', xlabel=None, ylabel = None)\n",
    "sn.countplot(x=\"OnlineBackup\", hue=\"Churn\",data=df, ax=axes[2,2]).set(title='Online Back up Services vs Churn Rates', xlabel=None, ylabel = None)\n",
    "sn.despine()\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "## 1) Prepare Data\n",
    "df_cluster = df.copy()\n",
    "df_cluster = df_cluster.drop(['customerID', 'TotalCharges'], axis=1)\n",
    "## Scale Tenure and Monthly Charges\n",
    "scaler = StandardScaler()\n",
    "df_cluster[['tenure', 'MonthlyCharges']] = scaler.fit_transform(df_cluster[['tenure', 'MonthlyCharges']])\n",
    "\n",
    "#Selecting all variables except tenure and Monthly Charges\n",
    "df_cluster[df_cluster.columns[~df_cluster.columns.isin(['tenure','MonthlyCharges'])]] = encoder.fit_transform(df_cluster[df_cluster.columns[~df_cluster.columns.isin(['tenure','MonthlyCharges'])]])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b97a4d74-d7b2-4c46-a776-89be5194c6ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 2) K-Means Clusters\n",
    "def optimise_k_means(data, max_k):\n",
    "    means = []\n",
    "    inertias = []\n",
    "    \n",
    "    for k in range(1,max_k):\n",
    "        kmeans = KMeans(n_clusters=k)\n",
    "        kmeans.fit(data)\n",
    "        means.append(k)\n",
    "        inertias.append(kmeans.inertia_)\n",
    "        \n",
    "    fig = plt.subplots(figsize=(10, 7))\n",
    "    plt.plot(means, inertias, 'o-', color = 'black')\n",
    "    plt.xlabel(\"Number of Clusters\")\n",
    "    plt.ylabel(\"Inertia\")\n",
    "    plt.grid(True)\n",
    "    plt.show()\n",
    "optimise_k_means(df_cluster, 10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e36ee04a-723f-4af9-a7a3-d7f56cc01b5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# K-Means cluster analysis\n",
    "kmeans = KMeans(n_clusters = 4,  random_state=10)\n",
    "kmeans.fit(df_cluster)\n",
    "# Save cluster group as a column value in our data_frame\n",
    "df_cluster['Cluster'] = kmeans.labels_\n",
    "# Cluster Counts\n",
    "fig, ax = plt.subplots(1,2, figsize=(18, 6))\n",
    "data_temp = df_cluster['Cluster'].value_counts().sort_index()\n",
    "ax[0].bar(data_temp.index, data_temp,\n",
    "          edgecolor='black', color=['#F5E8C7', '#ECCCB2', '#DEB6AB', '#AC7088']\n",
    "       ,width=0.55 )\n",
    "ax[0].set_title('Cluster Counts', loc='left', fontsize=19, fontweight='bold')\n",
    "for i in data_temp.index:\n",
    "    ax[0].annotate(f\"{data_temp[i]}\", \n",
    "                   xy=(i, data_temp[i] + 80),\n",
    "                   va = 'center', ha='center',fontweight='light', fontfamily='serif',\n",
    "                   color='black')\n",
    "for s in ['top', 'right']:\n",
    "    ax[0].spines[s].set_visible(False)\n",
    "    \n",
    "    \n",
    "sn.countplot(x='Cluster', hue=\"Churn\", palette=\"Greys\", data=df_cluster)\n",
    "sn.despine()\n",
    "plt.legend(title='', loc='upper left', labels=['No Churn', 'Churn'])\n",
    "plt.title(\"Cluster Vs Churn Rates\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1fabce8-96b2-4a7f-b2b0-084ad08ecc93",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Cluster'] = df_cluster['Cluster']\n",
    "sn.histplot(data=df, x=\"tenure\", hue=\"Cluster\", element=\"step\")\n",
    "plt.title('Tenure distribution by Cluster')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3f09389d-ee42-4981-9fe3-aa340c14eb7e",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,12))\n",
    "sn.despine()\n",
    "\n",
    "# Gray for No Churn, highlight Churn!\n",
    "colors = [\"#553939\", \"#808080\", \"#A27B5C\",\"#A9A9A9\"]\n",
    "# Set custom color palette\n",
    "sn.set_palette(sn.color_palette(colors))\n",
    "ax = sn.countplot(x=\"Contract\", hue=\"Cluster\", data=df, ax = axes[0,0]).set(title='Contracts by Cluster', xlabel=None, ylabel = None)\n",
    "ax = sn.countplot(x=\"SeniorCitizen\", hue=\"Cluster\", data=df, ax = axes[0,1]).set(title='SeniorCitizen by Cluster', xlabel=None, ylabel = None)\n",
    "ax = sn.countplot(y='InternetService', hue=\"Cluster\", data=df,ax = axes[1,0]).set(title='InternetService by Cluster', xlabel=None, ylabel = None)\n",
    "ax = sn.countplot(y='OnlineSecurity', hue=\"Cluster\", data=df, ax = axes[1,1]).set(title='OnlineSecurity by Cluster', xlabel=None, ylabel = None)\n",
    "sn.despine()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f8770c45-88a5-4b56-be45-c0e5225314f5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a9c8aa76-9713-4d04-8413-d6e0af2b1aaa",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b52a5494-0f34-4262-8c81-2230694b0f4b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b2a909f0-244f-41e0-9e6e-fe9b5b788cbe",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

