# Import the required libraries 
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.classifier import ClassificationReport
import plotly.express as px
import plotly.graph_objs as go
import random
import colorsys
from sklearn.metrics import classification_report

file = 'C:/Downloads/Crop_recommendation.csv'

# Load the dataset or create mock data if file not found
try:
    df = pd.read_csv(file)
except FileNotFoundError:
    df = pd.DataFrame({
        'N': np.random.randint(0, 140, 1000),
        'P': np.random.randint(0, 145, 1000),
        'K': np.random.randint(0, 205, 1000),
        'temperature': np.random.uniform(10, 40, 1000),
        'humidity': np.random.uniform(30, 100, 1000),
        'ph': np.random.uniform(4, 9, 1000),
        'rainfall': np.random.uniform(20, 300, 1000),
        'label': np.random.choice(['rice', 'wheat', 'maize', 'potato', 'cotton', 'sugar cane'], 1000)
    })

# Data Preprocessing
c = df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target'] = c.cat.codes
y = df.target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Adding SVM
svc_linear = SVC(kernel='linear').fit(X_train, y_train)
svc_poly = SVC(kernel='poly').fit(X_train, y_train)
svc_rbf = SVC(kernel='rbf').fit(X_train, y_train)
# Adding Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
# Adding Random Forest
rf_clf = RandomForestClassifier(max_depth=4, n_estimators=100, random_state=42).fit(X_train, y_train)
# Adding Gradient Boost
grad_clf = GradientBoostingClassifier().fit(X_train, y_train)

# Get detailed summary for each model
linear_report = classification_report(y_test, svc_linear.predict(X_test), target_names=df['label'].unique())
poly_report = classification_report(y_test, svc_poly.predict(X_test), target_names=df['label'].unique())
rbf_report = classification_report(y_test, svc_rbf.predict(X_test), target_names=df['label'].unique())
dt_report = classification_report(y_test, dt_clf.predict(X_test), target_names=df['label'].unique())
rf_report = classification_report(y_test, rf_clf.predict(X_test), target_names=df['label'].unique())
grad_report = classification_report(y_test, grad_clf.predict(X_test), target_names=df['label'].unique())


k_range = range(1, 11)
scores = []
# Adding KNN
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)  # Using X_train directly for KNN
    scores.append(knn.score(X_test, y_test))

# Visualize the results
def generate_crop_colors(n):

    hsv_colors = [(i/n, 0.9, 0.9) for i in range(n)]
    rgb_colors = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(*color)) for color in hsv_colors]
    hex_colors = [f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}' for color in rgb_colors]
    return hex_colors

# Generate the colors
num_crops = len(df['label'].unique())
crop_colors = generate_crop_colors(num_crops)

# Create a DataFrame for crop colors
crop_colors_df = pd.DataFrame({'label': df['label'].unique(), 'color': crop_colors})

# Create a dictionary to map each unique crop label to a specific color
crop_label_to_color = dict(zip(crop_colors_df['label'], crop_colors_df['color']))

# Add the 'target' column to the DataFrame
df['target'] = df['label'].map({crop: i for i, crop in enumerate(targets.values())})



# Mock Trends for each crop if csv not found
try:
    df = pd.read_csv(file)
except FileNotFoundError:
    for crop in crop_colors:
        if crop == 'rice':
            trend_value = np.linspace(0, 1, len(df[df['label'] == crop]))
            df.loc[df['label'] == crop, 'N'] += trend_value * 40
            df.loc[df['label'] == crop, 'P'] += trend_value * 30
            df.loc[df['label'] == crop, 'K'] += trend_value * 20
            df.loc[df['label'] == crop, 'temperature'] += trend_value * 3
            df.loc[df['label'] == crop, 'humidity'] -= trend_value * 5
            df.loc[df['label'] == crop, 'ph'] -= trend_value * 0.5
            df.loc[df['label'] == crop, 'rainfall'] += trend_value * 70

        elif crop == 'wheat':
            trend_value = np.linspace(0, 1, len(df[df['label'] == crop]))
            df.loc[df['label'] == crop, 'N'] += trend_value * 35
            df.loc[df['label'] == crop, 'P'] += trend_value * 28
            df.loc[df['label'] == crop, 'K'] += trend_value * 22
            df.loc[df['label'] == crop, 'temperature'] += trend_value * 4
            df.loc[df['label'] == crop, 'humidity'] -= trend_value * 4
            df.loc[df['label'] == crop, 'ph'] -= trend_value * 0.3
            df.loc[df['label'] == crop, 'rainfall'] += trend_value * 80
        
        elif crop == 'maize':
            trend_value = np.linspace(0, 1, len(df[df['label'] == crop]))
            df.loc[df['label'] == crop, 'N'] += trend_value * 50
            df.loc[df['label'] == crop, 'P'] += trend_value * 35
            df.loc[df['label'] == crop, 'K'] += trend_value * 25
            df.loc[df['label'] == crop, 'temperature'] += trend_value * 6
            df.loc[df['label'] == crop, 'humidity'] -= trend_value * 6
            df.loc[df['label'] == crop, 'ph'] -= trend_value * 0.8
            df.loc[df['label'] == crop, 'rainfall'] += trend_value * 90

        elif crop == 'potato':
            trend_value = np.linspace(0, 1, len(df[df['label'] == crop]))
            df.loc[df['label'] == crop, 'N'] += trend_value * 45
            df.loc[df['label'] == crop, 'P'] += trend_value * 40
            df.loc[df['label'] == crop, 'K'] += trend_value * 35
            df.loc[df['label'] == crop, 'temperature'] += trend_value * 2
            df.loc[df['label'] == crop, 'humidity'] -= trend_value * 8
            df.loc[df['label'] == crop, 'ph'] -= trend_value * 0.2
            df.loc[df['label'] == crop, 'rainfall'] += trend_value * 60

        elif crop == 'cotton':
            trend_value = np.linspace(0, 1, len(df[df['label'] == crop]))
            df.loc[df['label'] == crop, 'N'] += trend_value * 55
            df.loc[df['label'] == crop, 'P'] += trend_value * 25
            df.loc[df['label'] == crop, 'K'] += trend_value * 15
            df.loc[df['label'] == crop, 'temperature'] += trend_value * 8
            df.loc[df['label'] == crop, 'humidity'] -= trend_value * 2
            df.loc[df['label'] == crop, 'ph'] -= trend_value * 0.7
            df.loc[df['label'] == crop, 'rainfall'] += trend_value * 110

        elif crop == 'sugar cane':
            trend_value = np.linspace(0, 1, len(df[df['label'] == crop]))
            df.loc[df['label'] == crop, 'N'] += trend_value * 60
            df.loc[df['label'] == crop, 'P'] += trend_value * 30
            df.loc[df['label'] == crop, 'K'] += trend_value * 18
            df.loc[df['label'] == crop, 'temperature'] += trend_value * 5
            df.loc[df['label'] == crop, 'humidity'] -= trend_value * 3
            df.loc[df['label'] == crop, 'ph'] -= trend_value * 0.5
            df.loc[df['label'] == crop, 'rainfall'] += trend_value * 120

# create app
app = dash.Dash(__name__)

# create layout
app.layout = html.Div(children=[
    html.H1("Crop Recommendation Dashboard", style={'textAlign': 'center'}),
    dash_table.DataTable(
        id='summary-table',
        columns=[
            {'name': 'Statistics', 'id': 'index'},
            * [{'name': col, 'id': col} for col in df.describe().columns]
        ],
        data=round(df.describe(), 2).reset_index().to_dict('records'),  # Round the values to 2 decimal places and include row headers
        style_table={'width': '50%', 'margin': '20px auto'}
    ),
    html.H3("Linear Kernel Accuracy:"),
    dcc.Markdown(f"```\n{linear_report}\n```"),

    html.H3("Poly Kernel Accuracy:"),
    dcc.Markdown(f"```\n{poly_report}\n```"),

    html.H3("RBF Kernel Accuracy:"),
    dcc.Markdown(f"```\n{rbf_report}\n```"),

    html.H3("Decision Tree Accuracy:"),
    dcc.Markdown(f"```\n{dt_report}\n```"),

    html.H3("Random Forest Accuracy:"),
    dcc.Markdown(f"```\n{rf_report}\n```"),

    html.H3("Gradient Boosting Accuracy:"),
    dcc.Markdown(f"```\n{grad_report}\n```"),
    dcc.Graph(
        id='heatmap',
        figure={
            'data': [go.Heatmap(z=X.corr(), x=X.columns, y=X.columns, colorscale='Viridis')],
            'layout': {'title': 'Correlation Heatmap'}
        }
    ),

    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                go.Scatter(
                    x=df['K'],
                    y=df['N'],
                    mode='markers',
                    text=df['label'],
                    marker=dict(color=[crop_label_to_color[crop] for crop in df['label']])
                )
            ],
            'layout': {'title': 'Scatter Plot (K vs N)'}
        }
    ),

    dcc.Graph(
        id='count-plot',
        figure={
            'data': [
                {
                    'x': df['label'].value_counts().index,
                    'y': df['label'].value_counts().values,
                    'type': 'bar',
                    'marker': {'color': [crop_label_to_color[crop] for crop in df['label'].value_counts().index]}
                }
            ],
            'layout': {'title': 'Crop Counts'}
        }
    ),

    dcc.Graph(
        id='pair-plot',
        figure=px.scatter_matrix(df, dimensions=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], color='label', color_discrete_map=crop_label_to_color).update_layout(
            title='Pair Plot',
            xaxis=dict(title='Features'),
            yaxis=dict(title='Features')
        )
    ),

    dcc.Graph(
        id='hist-temperature',
        figure=go.Figure(data=go.Histogram(x=df['temperature'], marker=dict(color="purple"), nbinsx=15, opacity=0.2)).update_layout(
            title='Temperature Histogram',
            xaxis=dict(title='Temperature'),
            yaxis=dict(title='Count')
        )
    ),

    dcc.Graph(
        id='hist-ph',
        figure=go.Figure(data=go.Histogram(x=df['ph'], marker=dict(color="green"), nbinsx=15, opacity=0.2)).update_layout(
            title='pH Histogram',
            xaxis=dict(title='pH'),
            yaxis=dict(title='Count')
        )
    ),

    dcc.Graph(
        id='joint-rainfall-humidity',
        figure=go.Figure(data=go.Scatter(
            x=df[(df['temperature'] < 30) & (df['rainfall'] > 120)]['rainfall'],
            y=df[(df['temperature'] < 30) & (df['rainfall'] > 120)]['humidity'],
            mode='markers',
            text=df[(df['temperature'] < 30) & (df['rainfall'] > 120)]['label'],
            marker=dict(color=[crop_label_to_color[crop] for crop in df[(df['temperature'] < 30) & (df['rainfall'] > 120)]['label']])
        )).update_layout(
            title='Joint Plot of Rainfall vs Humidity',
            xaxis=dict(title='Rainfall'),
            yaxis=dict(title='Humidity'),
        )
    ),

    dcc.Graph(
        id='joint-k-n',
        figure=go.Figure(data=go.Scatter(
            x=df[(df['N'] > 40) & (df['K'] > 40)]['K'],
            y=df[(df['N'] > 40) & (df['K'] > 40)]['N'],
            mode='markers',
            text=df[(df['N'] > 40) & (df['K'] > 40)]['label'],
            marker=dict(color=[crop_label_to_color[crop] for crop in df[(df['N'] > 40) & (df['K'] > 40)]['label']])
        )).update_layout(
            title='Joint Plot of K vs N',
            xaxis=dict(title='K'),
            yaxis=dict(title='N')
        )
    ),

    dcc.Graph(
        id='joint-k-humidity',
        figure=go.Figure(data=go.Scatter(
            x=df['K'],
            y=df['humidity'],
            mode='markers',
            text=df['label'],
            marker=dict(color=[crop_label_to_color[crop] for crop in df['label']])
        )).update_layout(
            title='Joint Plot of K vs Humidity',
            xaxis=dict(title='K'),
            yaxis=dict(title='Humidity')
        )
    ),

    dcc.Graph(
        id='box-ph-crop',
        figure=go.Figure(data=[go.Box(
            y=df[df['label'] == crop]['ph'],
            name=crop,
            marker=dict(color=crop_label_to_color[crop])
        ) for crop in df['label'].unique()]).update_layout(
            title='Box Plot of pH vs Crop',
            xaxis=dict(title='Crop'),
            yaxis=dict(title='pH')
        )
    ),

    dcc.Graph(
        id='box-p-rainfall',
        figure=go.Figure(data=[
            go.Box(
                y=df[df['label'] == crop]['P'],
                name=crop,
                marker=dict(color=crop_label_to_color[crop])
            ) for crop in df['label'].unique() if df[df['label'] == crop]['rainfall'].mean() > 150
        ]).update_layout(
            title='Box Plot of P vs Rainfall',
            xaxis=dict(title='Crop'),
            yaxis=dict(title='P')
        )
    ),

    dcc.Graph(
        id='line-rainfall-k-humidity',
        figure=go.Figure(data=go.Scatter(
            x=df[(df['humidity'] < 65)]['K'],
            y=df[(df['humidity'] < 65)]['rainfall'],
            mode='lines+markers',
            text=df[(df['humidity'] < 65)]['label'],
            marker=dict(color=[crop_label_to_color[crop] for crop in df[(df['humidity'] < 65)]['label']])
        )).update_layout(
            title='Line Plot of Rainfall vs K with Humidity < 65',
            xaxis=dict(title='K'),
            yaxis=dict(title='Rainfall')
        )
    ),

    dcc.Graph(
        id='knn-accuracy-plot',
        figure=go.Figure(data=go.Scatter(x=list(k_range), y=scores, mode='markers+lines', marker=dict(color='blue'))).update_layout(
            title='KNN Accuracy Plot',
            xaxis=dict(title='K'),
            yaxis=dict(title='Accuracy')
        )
    )
,

   

])

if __name__ == '__main__':
    app.run_server(debug=True)
