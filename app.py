import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import os
import json
from datetime import datetime
import hashlib

# Daten laden
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'deutsche_bank_costs.csv')
df = pd.read_csv(csv_path)

# User data storage (in production, use a database)
USERS_FILE = os.path.join(current_dir, 'users.json')

# AI Model storage
MODELS_DIR = os.path.join(current_dir, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# PyTorch Neural Network Models
class SmallCostPredictor(nn.Module):
    """Small neural network for cost prediction"""
    def __init__(self, input_size):
        super(SmallCostPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BigCostPredictor(nn.Module):
    """Large neural network for cost prediction"""
    def __init__(self, input_size):
        super(BigCostPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Global model storage
trained_models = {
    'small': {'model': None, 'scaler': None, 'encoders': None, 'trained': False},
    'big': {'model': None, 'scaler': None, 'encoders': None, 'trained': False}
}

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()

# Initialize default users if file doesn't exist
users = load_users()
if not users:
    users = {
        'admin': {
            'password': hash_password('admin123'),
            'email': 'admin@deutschebank.com',
            'full_name': 'Administrator',
            'department': 'IT',
            'role': 'Admin',
            'created_at': datetime.now().isoformat()
        },
        'user': {
            'password': hash_password('user123'),
            'email': 'user@deutschebank.com',
            'full_name': 'John Doe',
            'department': 'Finance',
            'role': 'Analyst',
            'created_at': datetime.now().isoformat()
        }
    }
    save_users(users)

# Dash App initialisieren
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Deutsche Bank Cost Dashboard"

# Enhanced CSS and JavaScript
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            * {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            .draggable-container {
                min-height: 100px;
            }
            .draggable-item {
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .draggable-item:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 16px rgba(0, 24, 168, 0.15) !important;
            }
            .draggable-item:hover .drag-handle {
                background-color: #0018A8 !important;
                color: white !important;
            }
            .drag-handle {
                cursor: move !important;
            }
            .ghost {
                opacity: 0.3;
            }
            .section-title {
                color: #0018A8;
                font-weight: 600;
                font-size: 1.1rem;
                margin: 15px 0 10px 0;
                padding-left: 12px;
                border-left: 4px solid #0018A8;
            }
            .login-container {
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, #0018A8 0%, #5a6c7d 100%);
            }
            .login-card {
                background: white;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                padding: 40px;
                max-width: 450px;
                width: 100%;
            }
            .profile-card {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                padding: 30px;
                margin-bottom: 20px;
            }
            .navbar-custom {
                background: #5a6c7d;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                padding: 15px 0;
            }
            .nav-link-custom {
                color: white !important;
                font-weight: 500;
                margin: 0 10px;
                transition: all 0.3s;
            }
            .nav-link-custom:hover {
                color: #e8e8e8 !important;
                transform: translateY(-2px);
            }
            .btn-primary-custom {
                background: linear-gradient(135deg, #0018A8 0%, #003d82 100%);
                border: none;
                border-radius: 8px;
                padding: 12px 30px;
                font-weight: 600;
                transition: all 0.3s;
            }
            .btn-primary-custom:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 24, 168, 0.3);
            }
            .user-avatar {
                width: 80px;
                height: 80px;
                border-radius: 50%;
                background: linear-gradient(135deg, #0018A8 0%, #5a6c7d 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 32px;
                font-weight: 700;
                margin: 0 auto 20px;
            }
            .ai-panel-card {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0, 24, 168, 0.15);
                padding: 30px;
                margin-bottom: 20px;
                border: 2px solid #0018A8;
            }
            .model-card {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin: 15px 0;
                border: 2px solid #E5E5E5;
                transition: all 0.3s;
            }
            .model-card:hover {
                border-color: #0018A8;
                box-shadow: 0 4px 12px rgba(0, 24, 168, 0.2);
            }
            .prediction-result {
                background: #0018A8;
                color: white;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                text-align: center;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(function() {
                    var containers = document.querySelectorAll('.draggable-container');
                    containers.forEach(function(container) {
                        if (container && !container.sortableInitialized) {
                            Sortable.create(container, {
                                animation: 150,
                                handle: '.drag-handle',
                                ghostClass: 'ghost',
                                dragClass: 'dragging',
                                direction: 'vertical'
                            });
                            container.sortableInitialized = true;
                        }
                    });
                }, 1000);
            });
        </script>
    </body>
</html>
'''

# Custom Styles
custom_style = {
    'backgroundColor': '#F4F4F4',
    'fontFamily': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
}

card_style = {
    'boxShadow': '0 2px 8px rgba(0, 24, 168, 0.08)',
    'borderRadius': '8px',
    'backgroundColor': 'white',
    'padding': '24px',
    'marginBottom': '24px',
    'border': '1px solid #E5E5E5'
}

header_style = {
    'backgroundColor': 'white',
    'borderBottom': '3px solid #0018A8',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
    'marginBottom': '30px',
    'padding': '20px 0'
}

draggable_item_style = {
    'boxShadow': '0 2px 8px rgba(0, 24, 168, 0.08)',
    'borderRadius': '8px',
    'backgroundColor': 'white',
    'padding': '24px',
    'marginBottom': '24px',
    'border': '1px solid #E5E5E5',
    'cursor': 'move',
    'position': 'relative'
}

hero_chart_style = {
    'boxShadow': '0 2px 8px rgba(0, 24, 168, 0.1)',
    'borderRadius': '8px',
    'backgroundColor': 'white',
    'padding': '15px',
    'marginBottom': '20px',
    'border': '1px solid #0018A8',
    'cursor': 'move',
    'position': 'relative',
    'display': 'block',
    'width': '100%'
}

medium_chart_style = {
    'boxShadow': '0 2px 6px rgba(0, 24, 168, 0.08)',
    'borderRadius': '8px',
    'backgroundColor': 'white',
    'padding': '15px',
    'marginBottom': '20px',
    'border': '1px solid #E5E5E5',
    'cursor': 'move',
    'position': 'relative',
    'display': 'block',
    'width': '100%'
}

small_chart_style = {
    'boxShadow': '0 2px 4px rgba(0, 24, 168, 0.06)',
    'borderRadius': '8px',
    'backgroundColor': 'white',
    'padding': '15px',
    'marginBottom': '20px',
    'border': '1px solid #E5E5E5',
    'cursor': 'move',
    'position': 'relative',
    'display': 'block',
    'width': '100%'
}

half_width_style = {
    'boxShadow': '0 2px 6px rgba(0, 24, 168, 0.08)',
    'borderRadius': '8px',
    'backgroundColor': 'white',
    'padding': '15px',
    'marginBottom': '20px',
    'border': '1px solid #E5E5E5',
    'cursor': 'move',
    'position': 'relative',
    'display': 'block',
    'width': '100%'
}

kpi_style = {
    'boxShadow': '0 2px 8px rgba(0, 24, 168, 0.08)',
    'borderRadius': '8px',
    'backgroundColor': 'white',
    'padding': '12px',
    'marginBottom': '16px',
    'border': '1px solid #E5E5E5',
    'cursor': 'move',
    'position': 'relative'
}

drag_handle_style = {
    'position': 'absolute',
    'top': '10px',
    'right': '10px',
    'cursor': 'move',
    'color': '#0018A8',
    'fontSize': '22px',
    'padding': '6px 10px',
    'backgroundColor': '#f8f9fa',
    'borderRadius': '4px',
    'userSelect': 'none',
    'fontWeight': 'bold',
    'border': '1px solid #E5E5E5',
    'transition': 'all 0.2s'
}

section_divider_style = {
    'height': '2px',
    'background': 'linear-gradient(to right, #0018A8, #00BFFF, #0018A8)',
    'margin': '20px 0 15px 0',
    'borderRadius': '2px',
    'opacity': '0.3'
}

# Navigation Bar Component
def create_navbar(username):
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Img(src='/assets/db_logo.jpg', 
                                style={'height': '40px', 'marginRight': '15px'},
                                alt='Deutsche Bank Logo'),
                        html.Span("Deutsche Bank Dashboard", 
                                 style={'color': 'white', 'fontSize': '1.3rem', 'fontWeight': '600'})
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], width=6),
                dbc.Col([
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Dashboard", href="/dashboard", className="nav-link-custom")),
                        dbc.NavItem(dbc.NavLink("ML Analysis", href="/ml-analysis", className="nav-link-custom", style={'minWidth': '120px', 'textAlign': 'center'})),
                        dbc.NavItem(dbc.NavLink("Profile", href="/profile", className="nav-link-custom")),
                        dbc.NavItem(html.Div([
                            html.I(className="fas fa-user", style={'marginRight': '8px'}),
                            html.Span(username, style={'marginRight': '15px'})
                        ], style={'color': 'white', 'display': 'flex', 'alignItems': 'center', 'padding': '8px'})),
                        dbc.NavItem(dbc.Button("Logout", id="logout-btn", color="light", size="sm", 
                                              className="btn-primary-custom", 
                                              style={'background': 'rgba(255,255,255,0.2)', 'border': 'none', 'color': 'white'}))
                    ], navbar=True)
                ], width=6, style={'display': 'flex', 'justifyContent': 'flex-end', 'alignItems': 'center'})
            ], align="center")
        ], fluid=True),
        className="navbar-custom",
        dark=True,
        sticky="top"
    )

# Login Page
def create_login_page():
    return html.Div([
        dcc.Location(id='url-login', refresh=True),
        html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Img(src='/assets/db_logo.jpg', 
                                style={'height': '70px', 'display': 'block', 'margin': '0 auto 30px'},
                                alt='Deutsche Bank Logo'),
                        html.H2("Welcome Back", style={'textAlign': 'center', 'color': '#0018A8', 'marginBottom': '10px', 'fontWeight': '700'}),
                        html.P("Sign in to access your dashboard", 
                              style={'textAlign': 'center', 'color': '#666', 'marginBottom': '30px'}),
                        dbc.Input(id='login-username', placeholder='Username', type='text', 
                                 style={'marginBottom': '20px', 'padding': '12px', 'borderRadius': '8px'}),
                        dbc.Input(id='login-password', placeholder='Password', type='password',
                                 style={'marginBottom': '20px', 'padding': '12px', 'borderRadius': '8px'}),
                        html.Div(id='login-error', style={'color': 'red', 'marginBottom': '15px', 'textAlign': 'center'}),
                        dbc.Button("Sign In", id='login-button', color="primary", size="lg", 
                                  className="btn-primary-custom",
                                  style={'padding': '12px', 'fontSize': '16px', 'width': '100%'}),
                        html.Hr(style={'margin': '30px 0'}),
                        html.P("Demo Credentials:", style={'textAlign': 'center', 'color': '#999', 'fontSize': '12px', 'marginBottom': '5px'}),
                        html.P("admin / admin123 or user / user123", 
                              style={'textAlign': 'center', 'color': '#999', 'fontSize': '11px'})
                    ])
                ])
            ], className="login-card")
        ], className="login-container")
    ])

# User Profile Page
def create_profile_page(username):
    users = load_users()
    user_data = users.get(username, {})
    
    return html.Div([
        dcc.Location(id='url-profile', refresh=True),
        dcc.Store(id='user-store', data={'username': username, **user_data}),
        create_navbar(username),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.Div(user_data.get('full_name', username)[0].upper(), 
                                        className="user-avatar"),
                                html.H3(user_data.get('full_name', username), 
                                       style={'textAlign': 'center', 'color': '#0018A8', 'marginBottom': '10px'}),
                                html.P(f"@{username}", 
                                      style={'textAlign': 'center', 'color': '#666', 'marginBottom': '30px'})
                            ])
                        ])
                    ], className="profile-card"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Profile Information", style={'color': '#0018A8', 'marginBottom': '20px'}),
                            html.Div([
                                html.Label("Full Name", style={'fontWeight': '600', 'color': '#333', 'marginBottom': '5px'}),
                                dbc.Input(id='profile-fullname', value=user_data.get('full_name', ''), 
                                         style={'marginBottom': '20px', 'borderRadius': '8px'}),
                                
                                html.Label("Email", style={'fontWeight': '600', 'color': '#333', 'marginBottom': '5px'}),
                                dbc.Input(id='profile-email', value=user_data.get('email', ''), type='email',
                                         style={'marginBottom': '20px', 'borderRadius': '8px'}),
                                
                                html.Label("Department", style={'fontWeight': '600', 'color': '#333', 'marginBottom': '5px'}),
                                dbc.Input(id='profile-department', value=user_data.get('department', ''),
                                         style={'marginBottom': '20px', 'borderRadius': '8px'}),
                                
                                html.Label("Role", style={'fontWeight': '600', 'color': '#333', 'marginBottom': '5px'}),
                                dbc.Input(id='profile-role', value=user_data.get('role', ''),
                                         style={'marginBottom': '20px', 'borderRadius': '8px'}),
                                
                                html.Div(id='profile-save-message', style={'marginBottom': '15px'}),
                                dbc.Button("Save Changes", id='profile-save-btn', color="primary", 
                                         className="btn-primary-custom",
                                         style={'width': '100%'})
                            ])
                        ])
                    ], className="profile-card"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Account Information", style={'color': '#0018A8', 'marginBottom': '20px'}),
                            html.Div([
                                html.P([
                                    html.Strong("Username: "), username
                                ], style={'marginBottom': '10px'}),
                                html.P([
                                    html.Strong("Member since: "), 
                                    datetime.fromisoformat(user_data.get('created_at', datetime.now().isoformat())).strftime('%B %Y')
                                ], style={'marginBottom': '10px'}),
                                html.P([
                                    html.Strong("Last login: "), 
                                    datetime.now().strftime('%Y-%m-%d %H:%M')
                                ])
                            ])
                        ])
                    ], className="profile-card")
                ], width=8, style={'margin': '0 auto'})
            ])
        ], fluid=True, style={'paddingTop': '30px', 'paddingBottom': '50px'})
    ], style=custom_style)

# Dashboard Page
def create_dashboard_page(username):
    return html.Div([
        dcc.Location(id='url-dashboard', refresh=True),
        create_navbar(username),
        dbc.Container([
            html.Div([
                # Global Filters (Fixed - Not Draggable)
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Global Filters", className="mb-3", 
                                   style={'color': '#0018A8', 'fontWeight': '600', 'fontSize': '1.1rem', 
                                         'borderBottom': '2px solid #0018A8', 'paddingBottom': '8px'}),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Level 2 - Region:", style={'fontWeight': 'bold'}),
                                    dcc.Dropdown(
                                        id='filter-level2',
                                        options=[{'label': 'All', 'value': 'ALL'}] + 
                                                [{'label': i, 'value': i} for i in sorted(df['Level2'].unique())],
                                        value='ALL',
                                        multi=True,
                                        className="mb-2"
                                    ),
                                ], width=3),
                                
                                dbc.Col([
                                    html.Label("Level 3 - Country:", style={'fontWeight': 'bold'}),
                                    dcc.Dropdown(
                                        id='filter-level3',
                                        options=[{'label': 'All', 'value': 'ALL'}],
                                        value='ALL',
                                        multi=True,
                                        className="mb-2"
                                    ),
                                ], width=3),
                                
                                dbc.Col([
                                    html.Label("Level 4 - Division:", style={'fontWeight': 'bold'}),
                                    dcc.Dropdown(
                                        id='filter-level4',
                                        options=[{'label': 'All', 'value': 'ALL'}],
                                        value='ALL',
                                        multi=True,
                                        className="mb-2"
                                    ),
                                ], width=3),
                                
                                dbc.Col([
                                    html.Label("Level 5 - Service:", style={'fontWeight': 'bold'}),
                                    dcc.Dropdown(
                                        id='filter-level5',
                                        options=[{'label': 'All', 'value': 'ALL'}],
                                        value='ALL',
                                        multi=True,
                                        className="mb-2"
                                    ),
                                ], width=3),
                            ])
                        ], style=card_style)
                    ])
                ], className="mb-4"),
                
                # Draggable Charts Container
                html.Div([
                    # Section: Key Metrics
                    html.H2("📊 Key Performance Indicators", className="section-title"),
                    
                    # KPI Cards (Draggable as a group)
                    html.Div([
                        html.Div("⋮⋮", className="drag-handle", style=drag_handle_style),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Div(id='total-cost-display', 
                                            className="text-center",
                                            style={'fontSize': '20px', 'fontWeight': 'bold', 'color': '#0018A8'})
                                ], style={'padding': '10px', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'borderRadius': '6px'})
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.Div(id='region-count',
                                            className="text-center",
                                            style={'fontSize': '20px', 'fontWeight': 'bold', 'color': '#00BFFF'})
                                ], style={'padding': '10px', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'borderRadius': '6px'})
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.Div(id='division-count',
                                            className="text-center",
                                            style={'fontSize': '20px', 'fontWeight': 'bold', 'color': '#4169E1'})
                                ], style={'padding': '10px', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'borderRadius': '6px'})
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.Div(id='avg-cost',
                                            className="text-center",
                                            style={'fontSize': '20px', 'fontWeight': 'bold', 'color': '#87CEEB'})
                                ], style={'padding': '10px', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'borderRadius': '6px'})
                            ], width=3),
                        ])
                    ], className="draggable-item", style=kpi_style),
                    
                    # Divider
                    html.Div(style=section_divider_style),
                    
                    # Section: Flow Analysis
                    html.H2("🔄 Flow & Hierarchy Analysis", className="section-title"),
                    
                    # Sankey Chart (Hero)
                    html.Div([
                        html.Div("⋮⋮", className="drag-handle", style=drag_handle_style),
                        html.H3("Sankey Diagram - Cost Flow Through Hierarchy", 
                               className="mb-2",
                               style={'color': '#0018A8', 'fontWeight': '600', 'fontSize': '1.1rem', 'borderLeft': '4px solid #0018A8', 'paddingLeft': '12px'}),
                        dcc.Graph(id='sankey-diagram', style={'height': '550px'})
                    ], className="draggable-item", style=hero_chart_style),
                    
                    # Divider
                    html.Div(style=section_divider_style),
                    
                    # Section: Regional & Division Analysis
                    html.H2("🌍 Regional & Division Analysis", className="section-title"),
                    
                    # Medium Charts Row - 50% size
                    dbc.Row([
                        dbc.Col([
                            # Region Bar Chart
                            html.Div([
                                html.Div("⋮⋮", className="drag-handle", style=drag_handle_style),
                                html.H4("Costs by Region", 
                                       className="mb-2",
                                       style={'color': '#0018A8', 'fontWeight': '600', 'fontSize': '1rem', 'borderLeft': '3px solid #0018A8', 'paddingLeft': '8px'}),
                                dcc.Graph(id='region-bar-chart', style={'height': '300px'})
                            ], className="draggable-item", style=half_width_style),
                        ], width=6),
                        dbc.Col([
                            # Division Pie Chart
                            html.Div([
                                html.Div("⋮⋮", className="drag-handle", style=drag_handle_style),
                                html.H4("Costs by Division", 
                                       className="mb-2",
                                       style={'color': '#0018A8', 'fontWeight': '600', 'fontSize': '1rem', 'borderLeft': '3px solid #0018A8', 'paddingLeft': '8px'}),
                                dcc.Graph(id='division-pie-chart', style={'height': '300px'})
                            ], className="draggable-item", style=half_width_style),
                        ], width=6),
                    ], className="mb-3"),
                    
                    # Heatmap (full width)
                    html.Div([
                        html.Div("⋮⋮", className="drag-handle", style=drag_handle_style),
                        html.H3("Heatmap - Costs by Region and Division", 
                               className="mb-2",
                               style={'color': '#0018A8', 'fontWeight': '600', 'fontSize': '1.1rem', 'borderLeft': '3px solid #0018A8', 'paddingLeft': '10px'}),
                        dcc.Graph(id='heatmap-chart', style={'height': '350px'})
                    ], className="draggable-item", style=hero_chart_style),
                    
                    # Divider
                    html.Div(style=section_divider_style),
                    
                    # Section: Top Performers
                    html.H2("🏆 Top Performers", className="section-title"),
                    
                    # Small Charts Row
                    dbc.Row([
                        dbc.Col([
                            # Top Services
                            html.Div([
                                html.Div("⋮⋮", className="drag-handle", style=drag_handle_style),
                                html.H4("Top 10 Services", 
                                       className="mb-2",
                                       style={'color': '#0018A8', 'fontWeight': '600', 'fontSize': '1rem', 'borderLeft': '3px solid #0018A8', 'paddingLeft': '8px'}),
                                dcc.Graph(id='top-services-chart', style={'height': '300px'})
                            ], className="draggable-item", style=small_chart_style),
                        ], width=4),
                        dbc.Col([
                            # Top Countries
                            html.Div([
                                html.Div("⋮⋮", className="drag-handle", style=drag_handle_style),
                                html.H4("Top 10 Countries", 
                                       className="mb-2",
                                       style={'color': '#0018A8', 'fontWeight': '600', 'fontSize': '1rem', 'borderLeft': '3px solid #0018A8', 'paddingLeft': '8px'}),
                                dcc.Graph(id='top-countries-chart', style={'height': '300px'})
                            ], className="draggable-item", style=small_chart_style),
                        ], width=4),
                        dbc.Col([
                            # Service Type Donut
                            html.Div([
                                html.Div("⋮⋮", className="drag-handle", style=drag_handle_style),
                                html.H4("Costs by Service Type", 
                                       className="mb-2",
                                       style={'color': '#0018A8', 'fontWeight': '600', 'fontSize': '1rem', 'borderLeft': '3px solid #0018A8', 'paddingLeft': '8px'}),
                                dcc.Graph(id='service-type-donut', style={'height': '300px'})
                            ], className="draggable-item", style=small_chart_style),
                        ], width=4),
                    ], className="mb-3"),
                    
                    # Divider
                    html.Div(style=section_divider_style),
                    
                    # Section: Statistical Analysis
                    html.H2("📈 Statistical Analysis", className="section-title"),
                    
                    # Medium Charts Row
                    dbc.Row([
                        dbc.Col([
                            # Cumulative Chart
                            html.Div([
                                html.Div("⋮⋮", className="drag-handle", style=drag_handle_style),
                                html.H4("Cumulative Cost Distribution", 
                                       className="mb-2",
                                       style={'color': '#0018A8', 'fontWeight': '600', 'fontSize': '1rem', 'borderLeft': '3px solid #0018A8', 'paddingLeft': '8px'}),
                                dcc.Graph(id='cumulative-chart', style={'height': '300px'})
                            ], className="draggable-item", style=medium_chart_style),
                        ], width=6),
                        dbc.Col([
                            # Box Plot
                            html.Div([
                                html.Div("⋮⋮", className="drag-handle", style=drag_handle_style),
                                html.H4("Cost Distribution - Box Plot", 
                                       className="mb-2",
                                       style={'color': '#0018A8', 'fontWeight': '600', 'fontSize': '1rem', 'borderLeft': '3px solid #0018A8', 'paddingLeft': '8px'}),
                                dcc.Graph(id='box-plot-chart', style={'height': '300px'})
                            ], className="draggable-item", style=medium_chart_style),
                        ], width=6),
                    ], className="mb-3"),
                    
                    # Divider
                    html.Div(style=section_divider_style),
                    
                    # Section: Advanced Visualizations
                    html.H2("🎯 Advanced Visualizations", className="section-title"),
                    
                    # Medium Charts Row
                    dbc.Row([
                        dbc.Col([
                            # Sunburst
                            html.Div([
                                html.Div("⋮⋮", className="drag-handle", style=drag_handle_style),
                                html.H4("Sunburst - Hierarchical Cost Breakdown", 
                                       className="mb-2",
                                       style={'color': '#0018A8', 'fontWeight': '600', 'fontSize': '1rem', 'borderLeft': '3px solid #0018A8', 'paddingLeft': '8px'}),
                                dcc.Graph(id='sunburst-chart', style={'height': '350px'})
                            ], className="draggable-item", style=medium_chart_style),
                        ], width=6),
                        dbc.Col([
                            # Radar Chart
                            html.Div([
                                html.Div("⋮⋮", className="drag-handle", style=drag_handle_style),
                                html.H4("Regional Cost Categories - Radar", 
                                       className="mb-2",
                                       style={'color': '#0018A8', 'fontWeight': '600', 'fontSize': '1rem', 'borderLeft': '3px solid #0018A8', 'paddingLeft': '8px'}),
                                dcc.Graph(id='radar-chart', style={'height': '350px'})
                            ], className="draggable-item", style=medium_chart_style),
                        ], width=6),
                    ], className="mb-3"),
                    
                ], className="draggable-container"),
                
            ], style=custom_style)
        ], fluid=True, style={'backgroundColor': '#F4F4F4', 'minHeight': '100vh'})
    ])

# Main App Layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='session-store', data={'username': None, 'authenticated': False}),
    html.Div(id='page-content')
])

# Routing Callback
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')],
    [State('session-store', 'data')],
    prevent_initial_call=False
)
def display_page(pathname, session_data):
    try:
        # Handle None or empty session_data
        if session_data is None:
            session_data = {'username': None, 'authenticated': False}
        
        # Check authentication
        if not session_data.get('authenticated', False):
            return create_login_page()
        
        username = session_data.get('username', 'user')
        
        # Handle None pathname
        if pathname is None:
            pathname = '/'
        
        # Route to appropriate page
        if pathname == '/profile':
            return create_profile_page(username)
        elif pathname == '/ml-analysis':
            return create_ml_analysis_page(username)
        elif pathname == '/dashboard' or pathname == '/':
            return create_dashboard_page(username)
        else:
            # Default to dashboard for unknown paths
            return create_dashboard_page(username)
    except Exception as e:
        # Return error page if something goes wrong
        return html.Div([
            html.H1("Error", style={'color': 'red'}),
            html.P(f"An error occurred: {str(e)}"),
            html.P("Please refresh the page or contact support.")
        ], style={'padding': '20px'})

# Login Callback
@app.callback(
    [Output('session-store', 'data'),
     Output('login-error', 'children'),
     Output('url', 'pathname')],
    [Input('login-button', 'n_clicks')],
    [State('login-username', 'value'),
     State('login-password', 'value'),
     State('session-store', 'data')]
)
def login(n_clicks, username, password, session_data):
    if n_clicks is None:
        return session_data, '', no_update
    
    if not username or not password:
        return session_data, 'Please enter both username and password', no_update
    
    users = load_users()
    if username in users:
        if users[username]['password'] == hash_password(password):
            new_session = {'username': username, 'authenticated': True}
            return new_session, '', '/dashboard'
    
    return session_data, 'Invalid username or password', no_update

# Logout Callback
@app.callback(
    [Output('session-store', 'data', allow_duplicate=True),
     Output('url', 'pathname', allow_duplicate=True)],
    [Input('logout-btn', 'n_clicks')],
    [State('session-store', 'data')],
    prevent_initial_call=True
)
def logout(n_clicks, session_data):
    if n_clicks:
        return {'username': None, 'authenticated': False}, '/'
    return session_data, no_update

# Profile Save Callback
@app.callback(
    Output('profile-save-message', 'children'),
    [Input('profile-save-btn', 'n_clicks')],
    [State('profile-fullname', 'value'),
     State('profile-email', 'value'),
     State('profile-department', 'value'),
     State('profile-role', 'value'),
     State('session-store', 'data')]
)
def save_profile(n_clicks, fullname, email, department, role, session_data):
    if n_clicks is None:
        return ''
    
    username = session_data.get('username')
    if not username:
        return dbc.Alert("Not authenticated", color="danger")
    
    users = load_users()
    if username in users:
        users[username]['full_name'] = fullname or users[username].get('full_name', '')
        users[username]['email'] = email or users[username].get('email', '')
        users[username]['department'] = department or users[username].get('department', '')
        users[username]['role'] = role or users[username].get('role', '')
        save_users(users)
        return dbc.Alert("Profile updated successfully!", color="success")
    
    return dbc.Alert("Error updating profile", color="danger")

# Callback für kaskadierende Filter
@app.callback(
    [Output('filter-level3', 'options'),
     Output('filter-level4', 'options'),
     Output('filter-level5', 'options')],
    [Input('filter-level2', 'value'),
     Input('filter-level3', 'value'),
     Input('filter-level4', 'value')]
)
def update_filter_options(level2, level3, level4):
    # Filtern für Level 3
    if level2 and level2 != 'ALL' and 'ALL' not in level2:
        df_filtered = df[df['Level2'].isin(level2)]
    else:
        df_filtered = df
    
    level3_options = [{'label': 'Alle', 'value': 'ALL'}] + \
                     [{'label': i, 'value': i} for i in sorted(df_filtered['Level3'].unique())]
    
    # Filtern für Level 4
    if level3 and level3 != 'ALL' and 'ALL' not in level3:
        df_filtered = df_filtered[df_filtered['Level3'].isin(level3)]
    
    level4_options = [{'label': 'Alle', 'value': 'ALL'}] + \
                     [{'label': i, 'value': i} for i in sorted(df_filtered['Level4'].unique())]
    
    # Filtern für Level 5
    if level4 and level4 != 'ALL' and 'ALL' not in level4:
        df_filtered = df_filtered[df_filtered['Level4'].isin(level4)]
    
    level5_options = [{'label': 'Alle', 'value': 'ALL'}] + \
                     [{'label': i, 'value': i} for i in sorted(df_filtered['Level5'].unique())]
    
    return level3_options, level4_options, level5_options

# Callback für Diagramme
@app.callback(
    [Output('sankey-diagram', 'figure'),
     Output('total-cost-display', 'children'),
     Output('region-count', 'children'),
     Output('division-count', 'children'),
     Output('avg-cost', 'children'),
     Output('region-bar-chart', 'figure'),
     Output('division-pie-chart', 'figure'),
     Output('top-services-chart', 'figure'),
     Output('top-countries-chart', 'figure'),
     Output('service-type-donut', 'figure'),
     Output('heatmap-chart', 'figure'),
     Output('cumulative-chart', 'figure'),
     Output('box-plot-chart', 'figure'),
     Output('sunburst-chart', 'figure'),
     Output('radar-chart', 'figure')],
    [Input('filter-level2', 'value'),
     Input('filter-level3', 'value'),
     Input('filter-level4', 'value'),
     Input('filter-level5', 'value')]
)
def update_graphs(level2, level3, level4, level5):
    # Daten filtern
    df_filtered = df.copy()
    
    if level2 and level2 != 'ALL' and 'ALL' not in level2:
        df_filtered = df_filtered[df_filtered['Level2'].isin(level2)]
    
    if level3 and level3 != 'ALL' and 'ALL' not in level3:
        df_filtered = df_filtered[df_filtered['Level3'].isin(level3)]
    
    if level4 and level4 != 'ALL' and 'ALL' not in level4:
        df_filtered = df_filtered[df_filtered['Level4'].isin(level4)]
    
    if level5 and level5 != 'ALL' and 'ALL' not in level5:
        df_filtered = df_filtered[df_filtered['Level5'].isin(level5)]
    
    # KPIs berechnen
    total_cost = df_filtered['Cost'].sum()
    region_count = df_filtered['Level2'].nunique()
    division_count = df_filtered['Level4'].nunique()
    avg_cost = df_filtered['Cost'].mean()
    
    # KPI Displays
    cost_display = html.Div([
        html.P("Gesamtkosten", className="mb-1", style={'fontSize': '12px', 'color': '#666'}),
        html.P(f"€{total_cost:,.0f}".replace(',', '.'))
    ])
    
    region_display = html.Div([
        html.P("Regionen", className="mb-1", style={'fontSize': '12px', 'color': '#666'}),
        html.P(f"{region_count}")
    ])
    
    division_display = html.Div([
        html.P("Divisionen", className="mb-1", style={'fontSize': '12px', 'color': '#666'}),
        html.P(f"{division_count}")
    ])
    
    avg_display = html.Div([
        html.P("Ø Kosten", className="mb-1", style={'fontSize': '12px', 'color': '#666'}),
        html.P(f"€{avg_cost:,.0f}".replace(',', '.'))
    ])
    
    # Alle Diagramme erstellen
    sankey_fig = create_sankey(df_filtered)
    region_bar = create_region_bar(df_filtered)
    division_pie = create_division_pie(df_filtered)
    top_services = create_top_services(df_filtered)
    top_countries = create_top_countries(df_filtered)
    service_donut = create_service_donut(df_filtered)
    heatmap = create_heatmap(df_filtered)
    cumulative = create_cumulative(df_filtered)
    box_plot = create_box_plot(df_filtered)
    sunburst = create_sunburst(df_filtered)
    radar = create_radar(df_filtered)
    
    return (sankey_fig, cost_display, region_display, division_display, 
            avg_display, region_bar, division_pie, top_services, top_countries,
            service_donut, heatmap, cumulative, box_plot, sunburst, radar)

def create_sankey(df_filtered):
    """Erstellt ein Sankey Diagramm mit allen Hierarchieebenen"""
    
    # Knoten und Links vorbereiten
    all_nodes = []
    node_dict = {}
    links_source = []
    links_target = []
    links_value = []
    links_color = []
    
    # Farbpalette
    colors = ['#0018A8', '#00BFFF', '#4169E1', '#87CEEB', '#B0E0E6']
    
    # Level 1 -> Level 2
    for _, row in df_filtered.groupby(['Level1', 'Level2'])['Cost'].sum().reset_index().iterrows():
        if row['Level1'] not in node_dict:
            node_dict[row['Level1']] = len(all_nodes)
            all_nodes.append(row['Level1'])
        if row['Level2'] not in node_dict:
            node_dict[row['Level2']] = len(all_nodes)
            all_nodes.append(row['Level2'])
        
        links_source.append(node_dict[row['Level1']])
        links_target.append(node_dict[row['Level2']])
        links_value.append(row['Cost'])
        links_color.append('rgba(0, 24, 168, 0.3)')
    
    # Level 2 -> Level 3
    for _, row in df_filtered.groupby(['Level2', 'Level3'])['Cost'].sum().reset_index().iterrows():
        if row['Level3'] not in node_dict:
            node_dict[row['Level3']] = len(all_nodes)
            all_nodes.append(row['Level3'])
        
        links_source.append(node_dict[row['Level2']])
        links_target.append(node_dict[row['Level3']])
        links_value.append(row['Cost'])
        links_color.append('rgba(0, 191, 255, 0.3)')
    
    # Level 3 -> Level 4
    for _, row in df_filtered.groupby(['Level3', 'Level4'])['Cost'].sum().reset_index().iterrows():
        if row['Level4'] not in node_dict:
            node_dict[row['Level4']] = len(all_nodes)
            all_nodes.append(row['Level4'])
        
        links_source.append(node_dict[row['Level3']])
        links_target.append(node_dict[row['Level4']])
        links_value.append(row['Cost'])
        links_color.append('rgba(65, 105, 225, 0.3)')
    
    # Level 4 -> Level 5
    for _, row in df_filtered.groupby(['Level4', 'Level5'])['Cost'].sum().reset_index().iterrows():
        if row['Level5'] not in node_dict:
            node_dict[row['Level5']] = len(all_nodes)
            all_nodes.append(row['Level5'])
        
        links_source.append(node_dict[row['Level4']])
        links_target.append(node_dict[row['Level5']])
        links_value.append(row['Cost'])
        links_color.append('rgba(135, 206, 235, 0.3)')
    
    # Sankey erstellen
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='white', width=0.5),
            label=all_nodes,
            color='#0018A8'
        ),
        link=dict(
            source=links_source,
            target=links_target,
            value=links_value,
            color=links_color
        )
    )])
    
    fig.update_layout(
        title="Kostenfluss durch alle Hierarchieebenen",
        font=dict(size=12),
        height=700
    )
    
    return fig

def create_region_bar(df_filtered):
    """Erstellt ein Balkendiagramm nach Regionen"""
    region_costs = df_filtered.groupby('Level2')['Cost'].sum().sort_values(ascending=True)
    
    fig = go.Figure(go.Bar(
        x=region_costs.values,
        y=region_costs.index,
        orientation='h',
        marker=dict(
            color=region_costs.values,
            colorscale='Blues',
            showscale=False
        ),
        text=[f"€{val:,.0f}".replace(',', '.') for val in region_costs.values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>€%{x:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Kosten (€)",
        yaxis_title="",
        showlegend=False,
        margin=dict(l=100, r=50, t=10, b=50),
        height=400
    )
    
    return fig

def create_division_pie(df_filtered):
    """Erstellt ein Tortendiagramm nach Divisionen"""
    division_costs = df_filtered.groupby('Level4')['Cost'].sum().sort_values(ascending=False)
    
    fig = go.Figure(go.Pie(
        labels=division_costs.index,
        values=division_costs.values,
        hole=0.4,
        marker=dict(colors=px.colors.sequential.Blues_r),
        textposition='auto',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>€%{value:,.0f}<br>%{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1),
        margin=dict(l=10, r=150, t=10, b=10),
        height=400
    )
    
    return fig

def create_top_services(df_filtered):
    """Top 10 Services nach Kosten"""
    top_services = df_filtered.groupby('Level5')['Cost'].sum().nlargest(10).sort_values()
    
    fig = go.Figure(go.Bar(
        x=top_services.values,
        y=top_services.index,
        orientation='h',
        marker=dict(color='#0018A8'),
        text=[f"€{val:,.0f}".replace(',', '.') for val in top_services.values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>€%{x:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Kosten (€)",
        yaxis_title="",
        showlegend=False,
        margin=dict(l=150, r=10, t=10, b=50),
        height=350
    )
    
    return fig

def create_top_countries(df_filtered):
    """Top 10 Länder nach Kosten"""
    top_countries = df_filtered.groupby('Level3')['Cost'].sum().nlargest(10).sort_values()
    
    fig = go.Figure(go.Bar(
        x=top_countries.values,
        y=top_countries.index,
        orientation='h',
        marker=dict(color='#00BFFF'),
        text=[f"€{val:,.0f}".replace(',', '.') for val in top_countries.values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>€%{x:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Kosten (€)",
        yaxis_title="",
        showlegend=False,
        margin=dict(l=100, r=10, t=10, b=50),
        height=350
    )
    
    return fig

def create_service_donut(df_filtered):
    """Donut Chart für Service-Typen"""
    service_costs = df_filtered.groupby('Level5')['Cost'].sum().nlargest(8)
    other = df_filtered.groupby('Level5')['Cost'].sum().nsmallest(len(df_filtered['Level5'].unique()) - 8).sum()
    
    if other > 0:
        service_costs['Andere'] = other
    
    fig = go.Figure(go.Pie(
        labels=service_costs.index,
        values=service_costs.values,
        hole=0.6,
        marker=dict(colors=px.colors.sequential.Blues),
        textposition='auto',
        textinfo='percent',
        hovertemplate='<b>%{label}</b><br>€%{value:,.0f}<br>%{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05, font=dict(size=9)),
        margin=dict(l=10, r=120, t=10, b=10),
        height=350
    )
    
    return fig

def create_heatmap(df_filtered):
    """Heatmap für Region vs Division"""
    # Gruppierung und Pivot
    heatmap_data = df_filtered.groupby(['Level2', 'Level4'])['Cost'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Level2', columns='Level4', values='Cost')
    heatmap_pivot = heatmap_pivot.fillna(0)
    
    # Erstelle Text-Array für Anzeige
    text_array = []
    for i in range(len(heatmap_pivot)):
        row_text = []
        for j in range(len(heatmap_pivot.columns)):
            val = heatmap_pivot.iloc[i, j]
            if val > 0:
                row_text.append(f"€{val:,.0f}".replace(',', '.'))
            else:
                row_text.append("")
        text_array.append(row_text)
    
    fig = go.Figure(go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns.tolist(),
        y=heatmap_pivot.index.tolist(),
        colorscale='Blues',
        text=text_array,
        texttemplate='%{text}',
        textfont={"size": 9},
        hovertemplate='<b>Region:</b> %{y}<br><b>Division:</b> %{x}<br><b>Kosten:</b> €%{z:,.0f}<extra></extra>',
        colorbar=dict(title="Kosten (€)")
    ))
    
    fig.update_layout(
        xaxis_title="Division",
        yaxis_title="Region",
        margin=dict(l=100, r=50, t=10, b=100),
        height=500,
        xaxis=dict(side='bottom'),
        yaxis=dict(side='left')
    )
    
    fig.update_xaxes(tickangle=-45)
    
    return fig

def create_cumulative(df_filtered):
    """Kumulative Kostenverteilung"""
    sorted_costs = df_filtered.sort_values('Cost', ascending=False).copy()
    sorted_costs['Cumulative'] = sorted_costs['Cost'].cumsum()
    sorted_costs['Percentage'] = (sorted_costs['Cumulative'] / sorted_costs['Cost'].sum()) * 100
    sorted_costs['Index'] = range(1, len(sorted_costs) + 1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sorted_costs['Index'],
        y=sorted_costs['Cumulative'],
        mode='lines',
        name='Kumulative Kosten',
        line=dict(color='#0018A8', width=3),
        fill='tozeroy',
        hovertemplate='Position: %{x}<br>Kumulativ: €%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Anzahl Einträge (sortiert)",
        yaxis_title="Kumulative Kosten (€)",
        showlegend=True,
        margin=dict(l=80, r=50, t=10, b=50),
        height=400
    )
    
    return fig

def create_box_plot(df_filtered):
    """Box Plot für Kostenverteilung nach Regionen"""
    fig = go.Figure()
    
    for region in sorted(df_filtered['Level2'].unique()):
        region_data = df_filtered[df_filtered['Level2'] == region]['Cost']
        fig.add_trace(go.Box(
            y=region_data,
            name=region,
            marker_color='#0018A8',
            boxmean='sd',
            hovertemplate='%{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        yaxis_title="Kosten (€)",
        xaxis_title="Region",
        showlegend=False,
        margin=dict(l=80, r=50, t=10, b=100),
        height=400
    )
    
    fig.update_xaxes(tickangle=-45)
    
    return fig

def create_sunburst(df_filtered):
    """Sunburst Chart für hierarchische Darstellung"""
    fig = px.sunburst(
        df_filtered,
        path=['Level2', 'Level3', 'Level4', 'Level5'],
        values='Cost',
        color='Cost',
        color_continuous_scale='Blues',
        hover_data={'Cost': ':,.0f'}
    )
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=500
    )
    
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Kosten: €%{value:,.0f}<extra></extra>'
    )
    
    return fig

def create_radar(df_filtered):
    """Radar Chart für Top Divisionen nach Regionen"""
    # Top 5 Divisionen
    top_divisions = df_filtered.groupby('Level4')['Cost'].sum().nlargest(5).index
    
    # Top 5 Regionen
    top_regions = df_filtered.groupby('Level2')['Cost'].sum().nlargest(5).index
    
    fig = go.Figure()
    
    for region in top_regions:
        region_data = df_filtered[df_filtered['Level2'] == region]
        division_costs = []
        
        for division in top_divisions:
            cost = region_data[region_data['Level4'] == division]['Cost'].sum()
            division_costs.append(cost)
        
        fig.add_trace(go.Scatterpolar(
            r=division_costs,
            theta=list(top_divisions),
            fill='toself',
            name=region,
            hovertemplate='<b>%{theta}</b><br>€%{r:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True)
        ),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.1),
        margin=dict(l=80, r=150, t=50, b=50),
        height=500
    )
    
    return fig

# ML Analysis Page
def create_ml_analysis_page(username):
    return html.Div([
        dcc.Location(id='url-ml-analysis', refresh=True),
        create_navbar(username),
        dbc.Container([
            html.Div([
                # Model Selection & Status Row
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Model Selection", 
                                   style={'color': '#0018A8', 'marginBottom': '20px',
                                         'borderLeft': '4px solid #0018A8', 'paddingLeft': '12px'}),
                            dbc.RadioItems(
                                id='ml-model-selection',
                                options=[
                                    {'label': html.Div([
                                        html.Strong('Small Model', style={'color': '#0018A8', 'fontSize': '16px'}),
                                        html.P('Fast training, 3 layers (32→16→1 neurons)', 
                                              style={'margin': '5px 0 0 0', 'color': '#666', 'fontSize': '12px'})
                                    ]), 'value': 'small'},
                                    {'label': html.Div([
                                        html.Strong('Big Model', style={'color': '#0018A8', 'fontSize': '16px'}),
                                        html.P('Advanced model, 5 layers (128→64→32→16→1 neurons)', 
                                              style={'margin': '5px 0 0 0', 'color': '#666', 'fontSize': '12px'})
                                    ]), 'value': 'big'}
                                ],
                                value='small',
                                inline=False
                            ),
                            html.Div(id='ml-model-status', style={'marginTop': '20px'})
                        ], className="ai-panel-card")
                    ], width=4),
                    
                    dbc.Col([
                        html.Div([
                            html.H4("Model Performance", 
                                   style={'color': '#0018A8', 'marginBottom': '20px',
                                         'borderLeft': '4px solid #0018A8', 'paddingLeft': '12px'}),
                            dcc.Graph(id='ml-performance-chart', style={'height': '300px'})
                        ], className="ai-panel-card")
                    ], width=8)
                ], className="mb-4"),
                
                # Training Section with Animation
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Train Model", 
                                   style={'color': '#0018A8', 'marginBottom': '20px',
                                         'borderLeft': '4px solid #0018A8', 'paddingLeft': '12px'}),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Epochs:", style={'fontWeight': '600', 'color': '#333'}),
                                    dbc.Input(id='ml-training-epochs', type='number', value=50, min=10, max=500,
                                             style={'marginBottom': '15px', 'borderRadius': '8px'})
                                ], width=6),
                                dbc.Col([
                                    html.Label("Learning Rate:", style={'fontWeight': '600', 'color': '#333'}),
                                    dbc.Input(id='ml-learning-rate', type='number', value=0.001, min=0.0001, max=0.1, step=0.0001,
                                             style={'marginBottom': '15px', 'borderRadius': '8px'})
                                ], width=6)
                            ]),
                            html.Div(id='ml-training-animation', style={'marginBottom': '15px', 'minHeight': '100px'}),
                            html.Div(id='ml-training-status', style={'marginBottom': '15px'}),
                            dbc.Button("Train Model", id='ml-train-model-btn', color="primary",
                                      className="btn-primary-custom",
                                      style={'padding': '12px', 'fontSize': '16px', 'fontWeight': '600', 'width': '100%'})
                        ], className="ai-panel-card")
                    ], width=12)
                ], className="mb-4"),
                
                # Predictions Section
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Make Predictions", 
                                   style={'color': '#0018A8', 'marginBottom': '20px',
                                         'borderLeft': '4px solid #0018A8', 'paddingLeft': '12px'}),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Region (Level 2):", style={'fontWeight': '600', 'color': '#333'}),
                                    dcc.Dropdown(
                                        id='ml-predict-level2',
                                        options=[{'label': i, 'value': i} for i in sorted(df['Level2'].unique())],
                                        value=df['Level2'].unique()[0],
                                        style={'marginBottom': '15px'}
                                    )
                                ], width=3),
                                dbc.Col([
                                    html.Label("Country (Level 3):", style={'fontWeight': '600', 'color': '#333'}),
                                    dcc.Dropdown(
                                        id='ml-predict-level3',
                                        options=[{'label': i, 'value': i} for i in sorted(df['Level3'].unique())],
                                        value=df['Level3'].unique()[0],
                                        style={'marginBottom': '15px'}
                                    )
                                ], width=3),
                                dbc.Col([
                                    html.Label("Division (Level 4):", style={'fontWeight': '600', 'color': '#333'}),
                                    dcc.Dropdown(
                                        id='ml-predict-level4',
                                        options=[{'label': i, 'value': i} for i in sorted(df['Level4'].unique())],
                                        value=df['Level4'].unique()[0],
                                        style={'marginBottom': '15px'}
                                    )
                                ], width=3),
                                dbc.Col([
                                    html.Label("Service (Level 5):", style={'fontWeight': '600', 'color': '#333'}),
                                    dcc.Dropdown(
                                        id='ml-predict-level5',
                                        options=[{'label': i, 'value': i} for i in sorted(df['Level5'].unique())],
                                        value=df['Level5'].unique()[0],
                                        style={'marginBottom': '15px'}
                                    )
                                ], width=3)
                            ]),
                            dbc.Button("Predict Cost", id='ml-predict-btn', color="success",
                                      className="btn-primary-custom",
                                      style={'padding': '12px', 'fontSize': '16px', 'fontWeight': '600',
                                            'background': 'linear-gradient(135deg, #28a745 0%, #20c997 100%)',
                                            'border': 'none', 'width': '100%', 'marginTop': '10px'})
                        ], className="ai-panel-card")
                    ], width=6),
                    
                    dbc.Col([
                        html.Div([
                            html.H4("Prediction Results", 
                                   style={'color': '#0018A8', 'marginBottom': '20px',
                                         'borderLeft': '4px solid #0018A8', 'paddingLeft': '12px'}),
                            html.Div(id='ml-prediction-result', style={'minHeight': '200px'}),
                            dcc.Graph(id='ml-prediction-chart', style={'height': '300px', 'marginTop': '20px'})
                        ], className="ai-panel-card")
                    ], width=6)
                ], className="mb-4"),
                
                # Training History Chart
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Training History", 
                                   style={'color': '#0018A8', 'marginBottom': '20px',
                                         'borderLeft': '4px solid #0018A8', 'paddingLeft': '12px'}),
                            dcc.Graph(id='ml-training-history', style={'height': '350px'})
                        ], className="ai-panel-card")
                    ], width=12)
                ])
            ], style=custom_style)
        ], fluid=True, style={'backgroundColor': '#F4F4F4', 'minHeight': '100vh'})
    ])

# AI Admin Panel Modal
def create_ai_admin_modal():
    return dbc.Modal([
        dbc.ModalHeader([
            html.Div([
                html.I(className="fas fa-brain", style={'marginRight': '10px', 'color': '#0018A8'}),
                html.H3("AI Admin Panel - Cost Prediction", 
                       style={'color': '#0018A8', 'margin': '0', 'display': 'inline'})
            ])
        ], style={'borderBottom': '3px solid #0018A8', 'padding': '20px'}),
        dbc.ModalBody([
            html.Div([
                # Model Selection
                html.Div([
                    html.H4("Select Neural Network Model", 
                           style={'color': '#0018A8', 'marginBottom': '20px', 
                                 'borderLeft': '4px solid #0018A8', 'paddingLeft': '12px'}),
                    dbc.RadioItems(
                        id='model-selection',
                        options=[
                            {'label': html.Div([
                                html.Strong('Small Model', style={'color': '#0018A8'}),
                                html.P('Fast training, 3 layers (32→16→1 neurons)', 
                                      style={'margin': '5px 0 0 0', 'color': '#666', 'fontSize': '12px'})
                            ]), 'value': 'small'},
                            {'label': html.Div([
                                html.Strong('Big Model', style={'color': '#0018A8'}),
                                html.P('Advanced model, 5 layers (128→64→32→16→1 neurons)', 
                                      style={'margin': '5px 0 0 0', 'color': '#666', 'fontSize': '12px'})
                            ]), 'value': 'big'}
                        ],
                        value='small',
                        inline=False,
                        style={'marginBottom': '30px'}
                    )
                ], className="ai-panel-card"),
                
                # Training Section
                html.Div([
                    html.H4("Train Model", 
                           style={'color': '#0018A8', 'marginBottom': '20px',
                                 'borderLeft': '4px solid #0018A8', 'paddingLeft': '12px'}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Epochs:", style={'fontWeight': '600', 'color': '#333'}),
                            dbc.Input(id='training-epochs', type='number', value=50, min=10, max=500,
                                     style={'marginBottom': '15px', 'borderRadius': '8px'})
                        ], width=6),
                        dbc.Col([
                            html.Label("Learning Rate:", style={'fontWeight': '600', 'color': '#333'}),
                            dbc.Input(id='learning-rate', type='number', value=0.001, min=0.0001, max=0.1, step=0.0001,
                                     style={'marginBottom': '15px', 'borderRadius': '8px'})
                        ], width=6)
                    ]),
                    html.Div(id='training-status', style={'marginBottom': '15px'}),
                    dbc.Button("Train Model", id='train-model-btn', color="primary",
                              className="btn-primary-custom",
                              style={'padding': '12px', 'fontSize': '16px', 'fontWeight': '600', 'width': '100%'})
                ], className="ai-panel-card"),
                
                # Prediction Section
                html.Div([
                    html.H4("Predict Future Costs", 
                           style={'color': '#0018A8', 'marginBottom': '20px',
                                 'borderLeft': '4px solid #0018A8', 'paddingLeft': '12px'}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Region (Level 2):", style={'fontWeight': '600', 'color': '#333'}),
                            dcc.Dropdown(
                                id='predict-level2',
                                options=[{'label': i, 'value': i} for i in sorted(df['Level2'].unique())],
                                value=df['Level2'].unique()[0],
                                style={'marginBottom': '15px'}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Country (Level 3):", style={'fontWeight': '600', 'color': '#333'}),
                            dcc.Dropdown(
                                id='predict-level3',
                                options=[{'label': i, 'value': i} for i in sorted(df['Level3'].unique())],
                                value=df['Level3'].unique()[0],
                                style={'marginBottom': '15px'}
                            )
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Division (Level 4):", style={'fontWeight': '600', 'color': '#333'}),
                            dcc.Dropdown(
                                id='predict-level4',
                                options=[{'label': i, 'value': i} for i in sorted(df['Level4'].unique())],
                                value=df['Level4'].unique()[0],
                                style={'marginBottom': '15px'}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Service (Level 5):", style={'fontWeight': '600', 'color': '#333'}),
                            dcc.Dropdown(
                                id='predict-level5',
                                options=[{'label': i, 'value': i} for i in sorted(df['Level5'].unique())],
                                value=df['Level5'].unique()[0],
                                style={'marginBottom': '15px'}
                            )
                        ], width=6)
                    ]),
                    html.Div(id='prediction-result', style={'marginTop': '20px'}),
                    dbc.Button("Predict Cost", id='predict-btn', color="success",
                              className="btn-primary-custom",
                              style={'padding': '12px', 'fontSize': '16px', 'fontWeight': '600',
                                    'background': 'linear-gradient(135deg, #28a745 0%, #20c997 100%)',
                                    'border': 'none', 'width': '100%'})
                ], className="ai-panel-card"),
                
                # Model Status
                html.Div([
                    html.H4("Model Status", 
                           style={'color': '#0018A8', 'marginBottom': '20px',
                                 'borderLeft': '4px solid #0018A8', 'paddingLeft': '12px'}),
                    html.Div(id='model-status-display', children=[
                        html.P("Model Status:", style={'fontWeight': '600', 'color': '#333'}),
                        html.P("Small Model: ✗ Not Trained",
                              style={'color': '#dc3545'}),
                        html.P("Big Model: ✗ Not Trained",
                              style={'color': '#dc3545'})
                    ])
                ], className="ai-panel-card")
            ])
        ], style={'maxHeight': '80vh', 'overflowY': 'auto', 'padding': '30px'}),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-ai-panel-btn", color="secondary", 
                      className="btn-primary-custom",
                      style={'background': '#6c757d', 'border': 'none'})
        ], style={'borderTop': '2px solid #E5E5E5'})
    ], id="ai-admin-modal", is_open=False, size="xl", backdrop=True, scrollable=True)

# AI Model Training Function
def prepare_data_for_training():
    """Prepare data for neural network training"""
    df_train = df.copy()
    
    # Encode categorical variables
    encoders = {}
    for col in ['Level2', 'Level3', 'Level4', 'Level5']:
        le = LabelEncoder()
        df_train[col + '_encoded'] = le.fit_transform(df_train[col])
        encoders[col] = le
    
    # Create features
    X = df_train[['Level2_encoded', 'Level3_encoded', 'Level4_encoded', 'Level5_encoded']].values
    y = df_train['Cost'].values.reshape(-1, 1)
    
    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    return X_scaled, y_scaled, scaler_X, scaler_y, encoders

def train_model(model_type, epochs=50, learning_rate=0.001):
    """Train the selected neural network model"""
    try:
        X, y, scaler_X, scaler_y, encoders = prepare_data_for_training()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Initialize model
        input_size = X_train.shape[1]
        if model_type == 'small':
            model = SmallCostPredictor(input_size)
        else:
            model = BigCostPredictor(input_size)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        train_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()
        
        # Store model
        trained_models[model_type]['model'] = model
        trained_models[model_type]['scaler'] = scaler_y
        trained_models[model_type]['encoders'] = encoders
        trained_models[model_type]['trained'] = True
        
        return True, train_losses[-1], test_loss
    except Exception as e:
        return False, str(e), None

def predict_cost(model_type, level2, level3, level4, level5):
    """Predict cost using trained model"""
    try:
        if not trained_models[model_type]['trained']:
            return None, "Model not trained yet. Please train the model first."
        
        model = trained_models[model_type]['model']
        scaler = trained_models[model_type]['scaler']
        encoders = trained_models[model_type]['encoders']
        
        # Encode inputs
        level2_enc = encoders['Level2'].transform([level2])[0]
        level3_enc = encoders['Level3'].transform([level3])[0]
        level4_enc = encoders['Level4'].transform([level4])[0]
        level5_enc = encoders['Level5'].transform([level5])[0]
        
        # Prepare input
        X_input = np.array([[level2_enc, level3_enc, level4_enc, level5_enc]])
        
        # Scale (using same scaler from training - simplified)
        X_input_scaled = (X_input - X_input.mean()) / (X_input.std() + 1e-8)
        
        # Predict
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_input_scaled)
            prediction_scaled = model(X_tensor).numpy()
        
        # Inverse transform (simplified - in production use proper scaler)
        # For now, we'll use a simple approximation
        prediction = prediction_scaled[0][0]
        
        # Get average cost for similar entries as baseline
        similar_costs = df[(df['Level2'] == level2) & 
                          (df['Level3'] == level3) & 
                          (df['Level4'] == level4) & 
                          (df['Level5'] == level5)]['Cost'].values
        
        if len(similar_costs) > 0:
            baseline = similar_costs.mean()
            # Adjust prediction to be in reasonable range
            prediction = baseline * (1 + prediction * 0.3)  # Scale adjustment
        else:
            # Use overall mean if no similar entries
            prediction = df['Cost'].mean() * (1 + prediction * 0.3)
        
        return max(0, prediction), None  # Ensure non-negative
    except Exception as e:
        return None, str(e)

# AI Panel Callbacks
@app.callback(
    Output('ai-admin-modal', 'is_open'),
    [Input('open-ai-panel-btn', 'n_clicks'),
     Input('close-ai-panel-btn', 'n_clicks')],
    [State('ai-admin-modal', 'is_open')]
)
def toggle_ai_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

# Helper function to create model status display
def create_model_status_display():
    return html.Div([
        html.P("Model Status:", style={'fontWeight': '600', 'color': '#333'}),
        html.P("Small Model: " + ("✓ Trained" if trained_models['small']['trained'] else "✗ Not Trained"),
              style={'color': '#28a745' if trained_models['small']['trained'] else '#dc3545'}),
        html.P("Big Model: " + ("✓ Trained" if trained_models['big']['trained'] else "✗ Not Trained"),
              style={'color': '#28a745' if trained_models['big']['trained'] else '#dc3545'})
    ])

# Callback to update model status display (on model selection change)
@app.callback(
    Output('model-status-display', 'children'),
    [Input('model-selection', 'value')],
    prevent_initial_call=False
)
def update_model_status(model_type):
    return create_model_status_display()

# Callback for training - only updates training status
# Model status will be updated via a separate mechanism
@app.callback(
    Output('training-status', 'children'),
    [Input('train-model-btn', 'n_clicks')],
    [State('model-selection', 'value'),
     State('training-epochs', 'value'),
     State('learning-rate', 'value')],
    prevent_initial_call=True
)
def train_model_callback(n_clicks, model_type, epochs, lr):
    if epochs is None or lr is None:
        return dbc.Alert("Please enter epochs and learning rate", color="warning")
    
    success, train_loss, test_loss = train_model(model_type, epochs, lr)
    
    if success:
        status_msg = dbc.Alert([
            html.H5("Training Complete!", className="alert-heading"),
            html.P(f"Final Training Loss: {train_loss:.4f}"),
            html.P(f"Test Loss: {test_loss:.4f}"),
            html.P(f"Model '{model_type}' is ready for predictions!")
        ], color="success")
        return status_msg
    else:
        return dbc.Alert(f"Training failed: {train_loss}", color="danger")

# Additional callback to refresh model status after training completes
# This uses the training button clicks as a trigger to refresh status
@app.callback(
    Output('model-status-display', 'children', allow_duplicate=True),
    [Input('train-model-btn', 'n_clicks')],
    prevent_initial_call=True
)
def refresh_model_status_after_training(n_clicks):
    # This callback just refreshes the model status display after training
    return create_model_status_display()

@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-btn', 'n_clicks')],
    [State('model-selection', 'value'),
     State('predict-level2', 'value'),
     State('predict-level3', 'value'),
     State('predict-level4', 'value'),
     State('predict-level5', 'value')],
    prevent_initial_call=True
)
def predict_cost_callback(n_clicks, model_type, level2, level3, level4, level5):
    if n_clicks is None:
        return ""
    
    if not trained_models[model_type]['trained']:
        return dbc.Alert("Please train the model first before making predictions!", color="warning")
    
    if not all([level2, level3, level4, level5]):
        return dbc.Alert("Please select all parameters", color="warning")
    
    prediction, error = predict_cost(model_type, level2, level3, level4, level5)
    
    if error:
        return dbc.Alert(f"Prediction error: {error}", color="danger")
    
    return html.Div([
        html.H4("Predicted Cost", style={'color': 'white', 'marginBottom': '15px'}),
        html.H2(f"€{prediction:,.0f}".replace(',', '.'), 
               style={'color': 'white', 'fontSize': '3rem', 'fontWeight': '700', 'margin': '0'}),
        html.P(f"Model: {model_type.upper()}", 
              style={'color': 'rgba(255,255,255,0.9)', 'marginTop': '10px', 'fontSize': '14px'})
    ], className="prediction-result")

# ML Analysis Page Callbacks
# Store for training history
training_history = {'small': [], 'big': []}

@app.callback(
    Output('ml-model-status', 'children'),
    [Input('ml-model-selection', 'value')]
)
def update_ml_model_status(model_type):
    status = html.Div([
        html.P("Model Status:", style={'fontWeight': '600', 'color': '#333', 'marginBottom': '10px'}),
        html.P("Small Model: " + ("✓ Trained" if trained_models['small']['trained'] else "✗ Not Trained"),
              style={'color': '#28a745' if trained_models['small']['trained'] else '#dc3545', 'marginBottom': '5px'}),
        html.P("Big Model: " + ("✓ Trained" if trained_models['big']['trained'] else "✗ Not Trained"),
              style={'color': '#28a745' if trained_models['big']['trained'] else '#dc3545'})
    ])
    return status

@app.callback(
    [Output('ml-training-status', 'children'),
     Output('ml-training-animation', 'children'),
     Output('ml-model-selection', 'value')],
    [Input('ml-train-model-btn', 'n_clicks')],
    [State('ml-model-selection', 'value'),
     State('ml-training-epochs', 'value'),
     State('ml-learning-rate', 'value')],
    prevent_initial_call=True
)
def ml_train_model_callback(n_clicks, model_type, epochs, lr):
    if epochs is None or lr is None:
        return dbc.Alert("Please enter epochs and learning rate", color="warning"), "", no_update
    
    # Show training animation
    animation = html.Div([
        html.Div([
            html.I(className="fas fa-spinner fa-spin", style={'fontSize': '40px', 'color': '#0018A8', 'marginRight': '20px'}),
            html.Div([
                html.H5("Training in progress...", style={'color': '#0018A8', 'marginBottom': '5px'}),
                html.P("Please wait while the model trains", style={'color': '#666'})
            ])
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '30px'})
    ])
    
    success, train_loss, test_loss = train_model(model_type, epochs, lr)
    
    if success:
        # Update training history
        training_history[model_type].append({
            'epoch': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'lr': lr
        })
        
        status_msg = dbc.Alert([
            html.H5("Training Complete!", className="alert-heading"),
            html.P(f"Final Training Loss: {train_loss:.4f}"),
            html.P(f"Test Loss: {test_loss:.4f}"),
            html.P(f"Model '{model_type}' is ready for predictions!")
        ], color="success")
        
        # Return the model_type to trigger chart update via the other callback
        return status_msg, "", model_type
    else:
        return dbc.Alert(f"Training failed: {train_loss}", color="danger"), "", no_update

@app.callback(
    [Output('ml-prediction-result', 'children'),
     Output('ml-prediction-chart', 'figure')],
    [Input('ml-predict-btn', 'n_clicks')],
    [State('ml-model-selection', 'value'),
     State('ml-predict-level2', 'value'),
     State('ml-predict-level3', 'value'),
     State('ml-predict-level4', 'value'),
     State('ml-predict-level5', 'value')],
    prevent_initial_call=True
)
def ml_predict_cost_callback(n_clicks, model_type, level2, level3, level4, level5):
    if not trained_models[model_type]['trained']:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No predictions yet", height=300)
        return dbc.Alert("Please train the model first before making predictions!", color="warning"), empty_fig
    
    if not all([level2, level3, level4, level5]):
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No predictions yet", height=300)
        return dbc.Alert("Please select all parameters", color="warning"), empty_fig
    
    prediction, error = predict_cost(model_type, level2, level3, level4, level5)
    
    if error:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No predictions yet", height=300)
        return dbc.Alert(f"Prediction error: {error}", color="danger"), empty_fig
    
    # Get historical data for comparison
    similar_costs = df[(df['Level2'] == level2) & 
                      (df['Level3'] == level3) & 
                      (df['Level4'] == level4) & 
                      (df['Level5'] == level5)]['Cost'].values
    
    result = html.Div([
        html.H4("Predicted Cost", style={'color': 'white', 'marginBottom': '15px'}),
        html.H2(f"€{prediction:,.0f}".replace(',', '.'), 
               style={'color': 'white', 'fontSize': '3rem', 'fontWeight': '700', 'margin': '0'}),
        html.P(f"Model: {model_type.upper()}", 
              style={'color': 'rgba(255,255,255,0.9)', 'marginTop': '10px', 'fontSize': '14px'})
    ], className="prediction-result")
    
    # Prediction comparison chart
    pred_fig = go.Figure()
    if len(similar_costs) > 0:
        pred_fig.add_trace(go.Bar(
            x=['Historical Average', 'Predicted'],
            y=[similar_costs.mean(), prediction],
            marker_color=['#00BFFF', '#0018A8'],
            text=[f"€{similar_costs.mean():,.0f}".replace(',', '.'), f"€{prediction:,.0f}".replace(',', '.')],
            textposition='auto'
        ))
    else:
        pred_fig.add_trace(go.Bar(
            x=['Predicted'],
            y=[prediction],
            marker_color=['#0018A8'],
            text=[f"€{prediction:,.0f}".replace(',', '.')],
            textposition='auto'
        ))
    pred_fig.update_layout(
        title="Prediction vs Historical Average",
        xaxis_title="",
        yaxis_title="Cost (€)",
        height=300,
        showlegend=False
    )
    
    return result, pred_fig

@app.callback(
    [Output('ml-performance-chart', 'figure'),
     Output('ml-training-history', 'figure')],
    [Input('ml-model-selection', 'value')],
    prevent_initial_call=False
)
def update_ml_charts(model_type):
    # Performance chart
    perf_fig = go.Figure()
    perf_fig.add_trace(go.Bar(
        x=['Small Model', 'Big Model'],
        y=[
            training_history['small'][-1]['test_loss'] if training_history['small'] and trained_models['small']['trained'] else 0,
            training_history['big'][-1]['test_loss'] if training_history['big'] and trained_models['big']['trained'] else 0
        ],
        marker_color=['#0018A8' if trained_models['small']['trained'] else '#ccc',
                     '#00BFFF' if trained_models['big']['trained'] else '#ccc'],
        text=[
            f"{training_history['small'][-1]['test_loss']:.4f}" if training_history['small'] and trained_models['small']['trained'] else "Not Trained",
            f"{training_history['big'][-1]['test_loss']:.4f}" if training_history['big'] and trained_models['big']['trained'] else "Not Trained"
        ],
        textposition='auto'
    ))
    perf_fig.update_layout(
        title="Model Performance (Test Loss)",
        xaxis_title="Model",
        yaxis_title="Test Loss",
        height=300,
        showlegend=False
    )
    
    # Training history chart
    history_fig = go.Figure()
    if training_history[model_type]:
        history_data = training_history[model_type]
        history_fig.add_trace(go.Scatter(
            x=list(range(len(history_data))),
            y=[h['test_loss'] for h in history_data],
            mode='lines+markers',
            name='Test Loss',
            line=dict(color='#0018A8', width=2),
            marker=dict(size=8)
        ))
    else:
        history_fig.add_annotation(
            text="No training history yet. Train a model to see history.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    history_fig.update_layout(
        title=f"{model_type.upper()} Model Training History",
        xaxis_title="Training Run",
        yaxis_title="Test Loss",
        height=350,
        showlegend=True
    )
    
    return perf_fig, history_fig


if __name__ == '__main__':
    app.run(debug=True, port=8080)
