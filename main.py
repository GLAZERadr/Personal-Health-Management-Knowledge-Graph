import os
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
     

load_dotenv()

url=os.getenv("NEO4J_URI")
username=os.getenv("NEO4J_USERNAME")
password=os.getenv("NEO4J_PASSWORD")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API")

llm = ChatOpenAI(model="gpt-3.5-turbo")

driver = GraphDatabase.driver(uri=url, auth=(username, password))

def extract_entities(query):
    entities = {}

    # Extract user name (assuming it's provided beforehand)
    entities["user_name"] = "Adrian Putra Pratama Badjideh"  # Default or derived user name

    # Extract specific date (e.g., "October 15, 2024")
    date_match = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b', query)
    if date_match:
        date_str = date_match.group(0)
        try:
            specific_date = datetime.strptime(date_str, '%B %d, %Y')
            entities['date'] = specific_date.strftime('%Y-%m-%d')
        except ValueError:
            pass

    # Extract month (e.g., "June 2024")
    month_match = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b', query)
    if month_match:
        month_str = month_match.group(0)
        try:
            specific_month = datetime.strptime(month_str, '%B %Y')
            entities['month'] = specific_month.strftime('%Y-%m')
        except ValueError:
            pass

    # Extract activity type
    if "steps" in query or "physical activity" in query:
        entities['type'] = "physical_activity"
    elif "heart rate" in query or "blood pressure" in query:
        entities['type'] = "physiological_parameter"
    elif "sleep" in query:
        entities['type'] = "sleep_duration"
    elif "food" in query or "calories" in query:
        entities['type'] = "food"

    return entities



def retrieve_context(entities):
    try:
        context = []
        user_name = entities.get('user_name')
        if not user_name:
            raise ValueError("User name is required to retrieve context.")

        with driver.session() as session:
            if entities.get('type') == "physical_activity":
                print("Running query for User-Activity relationships...")
                time_condition = ""
                if 'date' in entities:
                    specific_date = datetime.strptime(entities['date'], '%Y-%m-%d')
                    epoch_time = int(specific_date.timestamp())
                    time_condition = f" AND a.timestamp >= {epoch_time} AND a.timestamp < {epoch_time + 86400}"
                elif 'month' in entities:
                    specific_month = datetime.strptime(entities['month'], '%Y-%m')
                    start_epoch = int(specific_month.timestamp())
                    next_month = specific_month.replace(day=28) + timedelta(days=4)  
                    next_month = next_month.replace(day=1)  
                    end_epoch = int(next_month.timestamp())
                    time_condition = f" AND a.timestamp >= {start_epoch} AND a.timestamp < {end_epoch}"

                activity_query = f"""
                MATCH (u:User)-[:HAS_DONE]->(a:PhysicalActivity)
                WHERE u.name = '{user_name}' {time_condition}
                RETURN u, a
                """
                print(f"Generated query: {activity_query}")

                result = session.run(activity_query)

                print("Raw data from User-Activity query:")
                found_data = False
                for record in result:
                    # print(record) 
                    found_data = True
                    timestamp_date = datetime.utcfromtimestamp(record['a']['timestamp']).strftime('%Y-%m-%d')
                    context.append(f"User {record['u']['name']} did {record['a']['daily_steps']} steps on {timestamp_date}.")

                if not found_data:
                    print("No data found for the query.")

            elif entities.get('type') == "physiological_parameter":
                print("Running query for User-PhysiologicalParameter relationships...")
                time_condition = ""
                if 'date' in entities:
                    specific_date = datetime.strptime(entities['date'], '%Y-%m-%d')
                    epoch_time = int(specific_date.timestamp())
                    time_condition = f" AND p.timestamp >= {epoch_time} AND p.timestamp < {epoch_time + 86400}"
                elif 'month' in entities:
                    specific_month = datetime.strptime(entities['month'], '%Y-%m')
                    start_epoch = int(specific_month.timestamp())
                    next_month = specific_month.replace(day=28) + timedelta(days=4)
                    next_month = next_month.replace(day=1)
                    end_epoch = int(next_month.timestamp())
                    time_condition = f" AND p.timestamp >= {start_epoch} AND p.timestamp < {end_epoch}"

                parameter_query = f"""
                MATCH (u:User)-[:HAS_MONITORED]->(p:PhysiologicalParameter)
                WHERE u.name = '{user_name}' {time_condition}
                RETURN u, p
                """
                print(f"Generated query: {parameter_query}")

                result = session.run(parameter_query)

                print("Raw data from User-PhysiologicalParameter query:")
                found_data = False
                for record in result:
                    # print(record)  
                    found_data = True
                    timestamp_date = datetime.utcfromtimestamp(record['p']['timestamp']).strftime('%Y-%m-%d')
                    context.append(f"User {record['u']['name']} had a heart rate of {record['p']['heart_rate']} bpm on {timestamp_date}.")

                if not found_data:
                    print("No data found for the query.")

            elif entities.get('type') == "sleep_duration":
                print("Running query for User-Sleep relationships...")
                time_condition = ""
                if 'date' in entities:
                    specific_date = datetime.strptime(entities['date'], '%Y-%m-%d')
                    epoch_time = int(specific_date.timestamp())
                    time_condition = f" AND s.timestamp >= {epoch_time} AND s.timestamp < {epoch_time + 86400}"
                elif 'month' in entities:
                    specific_month = datetime.strptime(entities['month'], '%Y-%m')
                    start_epoch = int(specific_month.timestamp())
                    next_month = specific_month.replace(day=28) + timedelta(days=4)
                    next_month = next_month.replace(day=1)
                    end_epoch = int(next_month.timestamp())
                    time_condition = f" AND s.timestamp >= {start_epoch} AND s.timestamp < {end_epoch}"

                sleep_query = f"""
                MATCH (u:User)-[:HAS_SLEEP]->(s:SleepDuration)
                WHERE u.name = '{user_name}' {time_condition}
                RETURN u, s
                """
                print(f"Generated query: {sleep_query}")

                result = session.run(sleep_query)

                print("Raw data from User-Sleep query:")
                found_data = False
                for record in result:
                    # print(record)  
                    found_data = True
                    timestamp_date = datetime.utcfromtimestamp(record['s']['timestamp']).strftime('%Y-%m-%d')
                    context.append(f"User {record['u']['name']} slept for {record['s']['hours']} hours and {record['s']['minutes']} minutes on {timestamp_date}.")

                if not found_data:
                    print("No data found for the query.")

            elif entities.get('type') == "food":
                print("Running query for User-Food relationships...")
                time_condition = ""
                if 'date' in entities:
                    specific_date = datetime.strptime(entities['date'], '%Y-%m-%d')
                    epoch_time = int(specific_date.timestamp())
                    time_condition = f" AND f.timestamp >= {epoch_time} AND f.timestamp < {epoch_time + 86400}"
                elif 'month' in entities:
                    specific_month = datetime.strptime(entities['month'], '%Y-%m')
                    start_epoch = int(specific_month.timestamp())
                    next_month = specific_month.replace(day=28) + timedelta(days=4)
                    next_month = next_month.replace(day=1)
                    end_epoch = int(next_month.timestamp())
                    time_condition = f" AND f.timestamp >= {start_epoch} AND f.timestamp < {end_epoch}"

                food_query = f"""
                MATCH (u:User)-[:HAS_CONSUMED]->(f:Food)
                WHERE u.name = '{user_name}' {time_condition}
                RETURN u, f
                """
                print(f"Generated query: {food_query}")

                result = session.run(food_query)

                print("Raw data from User-Food query:")
                found_data = False
                for record in result:
                    # print(record)  
                    found_data = True
                    timestamp_date = datetime.utcfromtimestamp(record['f']['timestamp']).strftime('%Y-%m-%d')
                    context.append(f"User {record['u']['name']} consumed {record['f']['name']} with {record['f']['calories']} calories on {timestamp_date}.")

                if not found_data:
                    print("No data found for the query.")

        return " ".join(context)

    except Exception as e:
        print(f"Error retrieving user context: {e}")
        return "An error occurred while retrieving user context."


# Initialize an empty list to store conversation history
conversation_history = []

def generate_response(query):
    # Extract entities to determine which relationships to query
    entities = extract_entities(query)

    # Retrieve context from the knowledge graph
    context = retrieve_context(entities)

    if not context or context == "An error occurred while retrieving user context.":
        context = "No specific data was found in the knowledge graph."

    # Few-shot examples to guide the LLM
    few_shot_examples = """
    Example 1:
    Data Available:
    User Adrian consumed 200 calories from Rice on April 1, 2024.
    User Adrian walked 5000 steps on April 1, 2024.
    Question: How many calories did Adrian consume on April 1, 2024?
    Answer: On April 1, 2024, Adrian consumed 200 calories.
    
    Example 2:
    Data Available:
    User Alex did 7000 steps on April 3, 2024.
    User Alex consumed 150 calories from Pasta on April 3, 2024.
    Question: How many steps did Alex take on April 3, 2024?
    Answer: On April 3, 2024, Alex did 7000 steps.
    """

    # Construct the conversation history prompt
    history_prompt = "\n".join([f"User: {q}\nAssistant: {r}" for q, r in conversation_history])

    # Combine few-shot examples with conversation history and the current context and user query
    prompt = f"""
    You are an assistant with access to user data. Use the following examples as a guide.

    {few_shot_examples}

    Conversation History:
    {history_prompt}

    Data Available:
    {context}

    User's Current Question: '{query}'
    """

    # Generate the response using the LLM
    response = llm(prompt)

    # Append the current query and response to conversation history
    conversation_history.append((query, response.content))

    return response.content


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://use.fontawesome.com/releases/v5.15.4/css/all.css"], suppress_callback_exceptions=True)
app.title = "Personal Health Management Knowledge Graph"
server = app.server

# Load data for the dashboard
dietary_habit_df = pd.read_csv('data/cleaned dietary habit.csv')
physical_activity_df = pd.read_csv('data/Physical Activity.csv')
physiological_parameter_df = pd.read_csv('data/Physiological Parameter.csv')
sleep_duration_df = pd.read_csv('data/Sleep Duration.csv')

COLORS = {
    'primary': '#3498db',   # Bright Blue
    'secondary': '#2ecc71', # Emerald Green
    'accent': '#e74c3c',    # Vibrant Red
    'background': '#f4f6f7' # Light Gray Background
}

# Sidebar layout for the dashboard
sidebar = html.Div(
    [
        html.H2("PHMKG", className="display-4", style={'padding': '20px', 'color': 'white'}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Health Dashboard", href="/dashboard", id="dashboard-link", style={'color': 'white'}),
                dbc.NavLink("Health Assistant Chatbot", href="/chatbot", id="chatbot-link", style={'color': 'white'})
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id='sidebar',
    style={
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'bottom': 0,
        'width': '18%',
        'padding': '20px',
        'backgroundColor': COLORS['primary'],
        'color': 'white',
        'transition': 'all 0.3s ease-in-out',
    },
)

# Sidebar toggle button
hidden_sidebar_style = {
    'position': 'fixed',
    'top': 0,
    'left': '-18%',  # Hidden state; shift sidebar off-screen
    'bottom': 0,
    'width': '18%',
    'padding': '20px',
    'backgroundColor': COLORS['primary'],
    'color': 'white',
    'transition': 'all 0.3s ease-in-out',
    'overflow': 'hidden'
}

visible_sidebar_style = {
    'position': 'fixed',
    'top': 0,
    'left': 0,  # Visible state
    'bottom': 0,
    'width': '18%',
    'padding': '20px',
    'backgroundColor': COLORS['primary'],
    'color': 'white',
    'transition': 'all 0.3s ease-in-out',
}

# Sidebar Toggle Button
sidebar_toggle_button = dbc.Button(
    html.I(className="fas fa-bars"), 
    id="sidebar-toggle", 
    color="secondary", 
    className="mb-3", 
    style={
        'position': 'fixed',
        'top': '10px',
        'left': '20px',
        'zIndex': 1000,
    }
)

# Enhanced Dashboard Layout
dashboard_layout = html.Div([
    sidebar,
    sidebar_toggle_button,
    html.Div(  # Added this main-content div
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Health Dashboard", className="text-center", style={'color': COLORS['primary'], 'marginTop': '20px'}),
                    html.Hr()
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    # Dietary Habits Card
                    dbc.Card([
                        dbc.CardHeader(html.H3("Dietary Habits", className="text-center", style={'color': COLORS['primary']})),
                        dbc.CardBody([
                            dcc.Graph(
                                id='calories-consumed',
                                figure=px.bar(dietary_habit_df, x='Date', y='Calories', color='Food_Name',
                                              title='Calories Consumed per Food Item',
                                              color_discrete_sequence=px.colors.qualitative.Set3)
                            )
                        ])
                    ], className="mb-4 shadow")
                ], width=12)
            ]),
            dbc.Row([
                # Combined Physical Activity Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H3("Physical Activity", className="text-center", 
                                               style={'color': COLORS['secondary']})),
                        dbc.CardBody([
                            dcc.Graph(
                                id='physical-activity',
                                figure=go.Figure(data=[
                                    go.Scatter(x=physical_activity_df['Timestamp'], 
                                               y=physical_activity_df['Daily Steps'], 
                                               mode='lines', 
                                               name='Daily Steps', 
                                               line=dict(color=COLORS['secondary'])),
                                    go.Scatter(x=physical_activity_df['Timestamp'], 
                                               y=physical_activity_df['Calories Burned'], 
                                               mode='lines', 
                                               name='Calories Burned', 
                                               line=dict(color=COLORS['accent']), 
                                               yaxis='y2')
                                ],
                                layout=go.Layout(
                                    title='Daily Steps and Calories Burned',
                                    xaxis=dict(title='Timestamp'),
                                    yaxis=dict(title='Daily Steps'),
                                    yaxis2=dict(title='Calories Burned', overlaying='y', side='right')
                                ))
                            )
                        ])
                    ], className="mb-4 shadow")
                ], width=12)
            ]),
            dbc.Row([
                # Combined Physiological Parameters Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H3("Physiological Parameters", className="text-center", 
                                               style={'color': COLORS['primary']})),
                        dbc.CardBody([
                            dcc.Graph(
                                id='physiological-parameters',
                                figure=go.Figure(data=[
                                    go.Scatter(x=physiological_parameter_df['Timestamp'], 
                                               y=physiological_parameter_df['Heart Rate'], 
                                               mode='lines', 
                                               name='Heart Rate', 
                                               line=dict(color=COLORS['primary'])),
                                    go.Scatter(x=physiological_parameter_df['Timestamp'], 
                                               y=physiological_parameter_df['Blood Oxygen'], 
                                               mode='lines', 
                                               name='Blood Oxygen', 
                                               line=dict(color=COLORS['accent']), 
                                               yaxis='y2')
                                ],
                                layout=go.Layout(
                                    title='Heart Rate and Blood Oxygen Levels',
                                    xaxis=dict(title='Timestamp'),
                                    yaxis=dict(title='Heart Rate'),
                                    yaxis2=dict(title='Blood Oxygen', overlaying='y', side='right')
                                ))
                            )
                        ])
                    ], className="mb-4 shadow")
                ], width=12)
            ]),
            dbc.Row([
                # Sleep Duration Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H3("Sleep Duration", className="text-center", 
                                               style={'color': COLORS['secondary']})),
                        dbc.CardBody([
                            dcc.Graph(
                                id='sleep-duration',
                                figure=px.bar(sleep_duration_df, x='Timestamp', y=['Hours', 'Minutes'],
                                              title='Sleep Duration (Hours and Minutes)',
                                              color_discrete_map={'Hours': COLORS['primary'], 'Minutes': COLORS['accent']})
                            )
                        ])
                    ], className="mb-4 shadow")
                ], width=12)
            ])
        ], fluid=True),
        id='main-content',  # This ID should be present in both layouts
        style={'marginLeft': '20%', 'transition': 'all 0.3s ease-in-out'}
    )
])

# Chatbot Layout
chatbot_layout = html.Div([
    sidebar,
    sidebar_toggle_button,
    html.Div(  # Added this main-content div
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Health Assistant Chatbot", className="text-center", 
                            style={'color': COLORS['primary'], 'marginTop': '20px'}),
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id='chatbot-messages', style={'height': '400px', 'overflowY': 'scroll'}),
                            dbc.InputGroup([
                                dbc.Input(id='user-input', placeholder='Type your health query...'),
                                dbc.Button('Send', id='send-button', color='primary')
                            ])
                        ])
                    ])
                ])
            ])
        ], fluid=True),
        id='main-content',  # This ID should be present in both layouts
        style={'marginLeft': '20%', 'transition': 'all 0.3s ease-in-out'}
    )
])

# App Layout with Multi-page support
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='sidebar-state', data=False),  # Initial state: sidebar is hidden
    html.Div(id='page-content')
])

@app.callback(
    [Output('sidebar', 'style'), Output('main-content', 'style'), Output('sidebar-state', 'data')],
    [Input('sidebar-toggle', 'n_clicks')],
    [State('sidebar-state', 'data')]
)
def toggle_sidebar(n_clicks, sidebar_open):
    if n_clicks is None:
        # Initial hidden state
        return hidden_sidebar_style, {'marginLeft': '2%', 'transition': 'all 0.3s ease-in-out'}, False

    # Toggle the sidebar's visibility state
    new_sidebar_open = not sidebar_open

    # Styles based on the new state of the sidebar
    if new_sidebar_open:
        sidebar_style = visible_sidebar_style
        main_content_style = {'marginLeft': '20%', 'transition': 'all 0.3s ease-in-out'}
    else:
        sidebar_style = hidden_sidebar_style
        main_content_style = {'marginLeft': '2%', 'transition': 'all 0.3s ease-in-out'}

    return sidebar_style, main_content_style, new_sidebar_open

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    # Render pages based on URL
    if pathname == '/dashboard':
        return dashboard_layout
    elif pathname == '/chatbot':
        return chatbot_layout
    else:
        return dashboard_layout  # Default to dashboard

# Optional: Add a basic chatbot callback (placeholder logic)
@app.callback(
    [Output('chatbot-messages', 'children'),
     Output('user-input', 'value')],
    [Input('send-button', 'n_clicks')],
    [State('user-input', 'value'),
     State('chatbot-messages', 'children')]
)
def update_chatbot(n_clicks, user_input, existing_messages):
    if not n_clicks or not user_input:
        raise dash.exceptions.PreventUpdate

    if not existing_messages:
        existing_messages = []

    # Generate the response using the `generate_response` function
    try:
        response = generate_response(user_input)
    except Exception as e:
        response = f"An error occurred: {e}"

    # Append the user's query and the bot's response to the chat history
    existing_messages.extend([
        html.Div(f"You: {user_input}", style={'marginBottom': '10px'}),
        html.Div(f"Bot: {response}", style={'marginBottom': '10px', 'color': COLORS['primary']})
    ])

    # Clear the input field after sending
    return existing_messages, ''


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
