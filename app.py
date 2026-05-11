import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier | AI-Powered Plant Identification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with proper contrast
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main background - light and clean */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
    }

    /* Main content area - white for better contrast */
    .main > div {
        background: white;
        border-radius: 30px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }

    /* Typography with dark colors for readability */
    h1 {
        color: #1a202c;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
        background: none;
        -webkit-text-fill-color: #1a202c;
    }

    h2 {
        color: #2d3748;
        font-weight: 700;
        background: none;
        -webkit-text-fill-color: #2d3748;
    }

    h3 {
        color: #4a5568;
        font-weight: 600;
        background: none;
        -webkit-text-fill-color: #4a5568;
    }

    p, li, span, div {
        color: #2d3748;
    }

    /* Sidebar styling - dark theme for contrast */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
        border-radius: 0 30px 30px 0;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: white !important;
        background: none;
        -webkit-text-fill-color: white;
    }

    section[data-testid="stSidebar"] .stMetric {
        background: rgba(255,255,255,0.15);
        border-radius: 15px;
        padding: 10px;
    }

    section[data-testid="stSidebar"] .stMetric label {
        color: white !important;
    }

    section[data-testid="stSidebar"] .stMetric div {
        color: white !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 25px;
        border: none;
        padding: 12px 28px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 16px;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 30px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        margin: 1rem 0;
        animation: slideIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        border: 1px solid rgba(255,255,255,0.2);
    }

    .prediction-card h1, 
    .prediction-card h2, 
    .prediction-card h3, 
    .prediction-card p {
        color: white !important;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(30px) scale(0.9);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        border-color: #667eea;
    }

    .metric-card h3 {
        color: #2d3748;
        margin: 0.5rem 0;
    }

    .metric-card p {
        color: #4a5568;
    }

    /* Feature cards */
    .feature-card {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    .feature-card strong {
        color: #2d3748;
    }

    .feature-card:hover {
        transform: translateX(5px);
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        background: white;
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .stSlider label {
        color: #2d3748 !important;
        font-weight: 500;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .stProgress label {
        color: #2d3748 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        border-radius: 50px;
        padding: 5px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 8px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        color: #4a5568;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }

    /* Info box */
    .info-box {
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
        border-left: 4px solid #667eea;
        background: #f7fafc;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    /* Alert styling */
    .stAlert {
        border-radius: 15px;
        border-left: 4px solid #667eea;
        background: #f7fafc;
        color: #2d3748;
    }

    .stAlert div {
        color: #2d3748;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f7fafc;
        border-radius: 15px;
        font-weight: 600;
        color: #2d3748;
    }

    .streamlit-expanderHeader:hover {
        background: #edf2f7;
    }

    /* Dataframe styling */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
    }

    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px;
    }

    .dataframe td {
        color: #2d3748;
        background: white;
    }

    /* Caption styling */
    .stCaption {
        color: #718096;
    }

    /* Success/Warning/Info boxes */
    .stSuccess {
        background: #c6f6d5;
        color: #22543d;
        border-radius: 15px;
    }

    .stWarning {
        background: #feebc8;
        color: #7b341e;
        border-radius: 15px;
    }

    .stInfo {
        background: #bee3f8;
        color: #2c5282;
        border-radius: 15px;
    }

    /* Number input styling */
    .stNumberInput input {
        color: #2d3748;
        background: white;
        border: 1px solid #e2e8f0;
    }

    /* Selectbox styling */
    .stSelectbox label {
        color: #2d3748 !important;
    }

    /* Markdown text styling */
    .stMarkdown p {
        color: #2d3748;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = iris.target_names[iris.target]
    return df, iris.target_names, iris.feature_names


@st.cache_resource
def train_model():
    df, _, _ = load_data()
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=4, min_samples_split=5)
    model.fit(df.iloc[:, :-2], df['species'])
    return model


# Load data and model
df, target_names, feature_names = load_data()
model = train_model()

# Calculate model accuracy
predictions = model.predict(df.iloc[:, :-2])
accuracy = accuracy_score(df['species'], predictions) * 100

# Sidebar
with st.sidebar:
    st.markdown("## 🌸 **Iris Classifier**")
    st.markdown("---")

    # Animated welcome
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <div style='font-size: 3rem; animation: pulse 2s infinite;'>🤖</div>
        <p style='margin-top: 0.5rem; opacity: 0.9;'>AI-Powered Plant ID</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model info
    st.markdown("### 🎯 **Performance**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.1f}%", delta="±2%")
    with col2:
        st.metric("Samples", f"{len(df)}", delta="flowers")

    st.markdown("---")

    # Feature importance
    st.markdown("### 📊 **Key Features**")
    importance = model.feature_importances_
    feature_names_imp = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    for name, imp in zip(feature_names_imp, importance):
        st.progress(float(imp), text=f"{name}: {imp * 100:.1f}%")

    st.markdown("---")

    # Quick tips
    st.markdown("### 💡 **Quick Tips**")
    st.info("""
    ✨ **Petal measurements** are most important
    🎯 **High confidence** >80% is reliable
    🎲 Try **Random Sample** for inspiration
    """)

    st.markdown("---")
    st.caption("Made with ❤️ • Streamlit")

# Header
col_title1, col_title2, col_title3 = st.columns([1, 2, 1])
with col_title2:
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <div style='font-size: 4rem; animation: pulse 2s infinite;'>🌸</div>
        <h1>Iris Flower Classifier</h1>
        <p style='font-size: 1.1rem; color: #4a5568; margin-top: 0.5rem;'>AI-Powered Plant Species Identification System</p>
        <p style='font-size: 0.9rem; color: #718096;'>Powered by Random Forest | 95%+ Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 **Predictor**", "📊 **Species Guide**", "📈 **Statistics**"])

with tab1:
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown("### 🎨 **Flower Measurements**")
        st.markdown("Adjust the sliders below:")

        # Session state
        if 'sl_sepal_length' not in st.session_state:
            st.session_state.sl_sepal_length = float(df['sepal length (cm)'].mean())
        if 'sl_sepal_width' not in st.session_state:
            st.session_state.sl_sepal_width = float(df['sepal width (cm)'].mean())
        if 'sl_petal_length' not in st.session_state:
            st.session_state.sl_petal_length = float(df['petal length (cm)'].mean())
        if 'sl_petal_width' not in st.session_state:
            st.session_state.sl_petal_width = float(df['petal width (cm)'].mean())

        sepal_length = st.slider(
            "📏 **Sepal Length** (cm)",
            min_value=float(df['sepal length (cm)'].min()),
            max_value=float(df['sepal length (cm)'].max()),
            value=st.session_state.sl_sepal_length,
            help="Length of the sepal from base to apex",
            key="sl_sepal_length_slider"
        )

        sepal_width = st.slider(
            "📐 **Sepal Width** (cm)",
            min_value=float(df['sepal width (cm)'].min()),
            max_value=float(df['sepal width (cm)'].max()),
            value=st.session_state.sl_sepal_width,
            help="Width of the sepal at its widest point",
            key="sl_sepal_width_slider"
        )

        petal_length = st.slider(
            "🌿 **Petal Length** (cm)",
            min_value=float(df['petal length (cm)'].min()),
            max_value=float(df['petal length (cm)'].max()),
            value=st.session_state.sl_petal_length,
            help="Length of the petal from base to tip",
            key="sl_petal_length_slider"
        )

        petal_width = st.slider(
            "🍃 **Petal Width** (cm)",
            min_value=float(df['petal width (cm)'].min()),
            max_value=float(df['petal width (cm)'].max()),
            value=st.session_state.sl_petal_width,
            help="Width of the petal at its widest point",
            key="sl_petal_width_slider"
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_button = st.button("🔮 **Predict Species**", use_container_width=True)
        with col_btn2:
            if st.button("🎲 **Random Sample**", use_container_width=True):
                random_row = df.sample(1).iloc[0]
                st.session_state.sl_sepal_length = float(random_row['sepal length (cm)'])
                st.session_state.sl_sepal_width = float(random_row['sepal width (cm)'])
                st.session_state.sl_petal_length = float(random_row['petal length (cm)'])
                st.session_state.sl_petal_width = float(random_row['petal width (cm)'])
                st.rerun()

    with col2:
        st.markdown("### 🌟 **Live Visualization**")

        current_values = [sepal_length, sepal_width, petal_length, petal_width]
        categories = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=current_values,
            theta=categories,
            fill='toself',
            name='Your Flower',
            line_color='#FF6B6B',
            fillcolor='rgba(255, 107, 107, 0.3)',
            line_width=3,
            hovertemplate='<b>Your Flower</b><br>%{theta}: %{r:.2f} cm<extra></extra>'
        ))

        avg_setosa = df[df['species_name'] == 'setosa'][feature_names].mean().values
        avg_versicolor = df[df['species_name'] == 'versicolor'][feature_names].mean().values
        avg_virginica = df[df['species_name'] == 'virginica'][feature_names].mean().values

        fig.add_trace(go.Scatterpolar(
            r=avg_setosa,
            theta=categories,
            name='Setosa (Avg)',
            line_color='#4ECDC4',
            line_dash='dash',
            line_width=2,
            hovertemplate='<b>Setosa Average</b><br>%{theta}: %{r:.2f} cm<extra></extra>'
        ))

        fig.add_trace(go.Scatterpolar(
            r=avg_versicolor,
            theta=categories,
            name='Versicolor (Avg)',
            line_color='#45B7D1',
            line_dash='dash',
            line_width=2,
            hovertemplate='<b>Versicolor Average</b><br>%{theta}: %{r:.2f} cm<extra></extra>'
        ))

        fig.add_trace(go.Scatterpolar(
            r=avg_virginica,
            theta=categories,
            name='Virginica (Avg)',
            line_color='#96CEB4',
            line_dash='dash',
            line_width=2,
            hovertemplate='<b>Virginica Average</b><br>%{theta}: %{r:.2f} cm<extra></extra>'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 8],
                    gridcolor='rgba(0,0,0,0.1)',
                    gridwidth=1,
                    tickfont=dict(size=12, color='#1a202c', weight='bold'),
                    title=dict(text='Centimeters', font=dict(size=12, color='#1a202c', weight='bold'))
                ),
                angularaxis=dict(
                    gridcolor='rgba(0,0,0,0.1)',
                    linecolor='rgba(0,0,0,0.2)',
                    tickfont=dict(size=12, color='#1a202c', weight='bold')
                ),
                bgcolor='white'
            ),
            showlegend=True,
            height=450,
            margin=dict(l=80, r=80, t=20, b=20),
            legend=dict(
                x=0.1,
                y=1.1,
                orientation='h',
                bgcolor='white',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1,
                font=dict(size=11, color='#1a202c', weight='bold')
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12, color='#1a202c')
        )

        st.plotly_chart(fig, use_container_width=True)

    if predict_button:
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        predict_species = target_names[prediction][0]
        confidence = np.max(prediction_proba) * 100

        st.markdown("---")
        st.markdown("### ✨ **Prediction Result**")

        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        with col_res2:
            emoji = '🌸' if predict_species == 'setosa' else '💜' if predict_species == 'versicolor' else '🌼'
            st.markdown(f"""
            <div class="prediction-card">
                <div style="font-size: 3rem; animation: pulse 2s infinite;">{emoji}</div>
                <h2 style="color: white; margin: 0.5rem 0;">{predict_species.title()}</h2>
                <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{confidence:.1f}%</div>
                <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 0.5rem; margin-top: 1rem;">
                    <p style="margin: 0;">🎯 Confidence Level: {'🔥 High' if confidence > 80 else '📊 Medium' if confidence > 60 else '⚠️ Low'}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 **Confidence Breakdown**")

        fig_conf = go.Figure(data=[
            go.Bar(
                x=target_names,
                y=prediction_proba[0] * 100,
                marker_color=['#4ECDC4', '#45B7D1', '#96CEB4'],
                text=[f'{x:.1f}%' for x in prediction_proba[0] * 100],
                textposition='auto',
                textfont=dict(size=14, color='#1a202c', weight='bold'),
                name='Confidence',
                hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
            )
        ])

        fig_conf.update_layout(
            title=dict(text="Prediction Confidence by Species", font=dict(size=18, color='#1a202c', weight='bold')),
            yaxis_title=dict(text="Confidence (%)", font=dict(size=14, color='#1a202c', weight='bold')),
            xaxis_title=dict(text="Species", font=dict(size=14, color='#1a202c', weight='bold')),
            yaxis_range=[0, 100],
            showlegend=False,
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12, color='#1a202c'),
            xaxis=dict(tickfont=dict(size=12, color='#1a202c', weight='bold')),
            yaxis=dict(tickfont=dict(size=12, color='#1a202c', weight='bold'), gridcolor='rgba(0,0,0,0.1)')
        )

        fig_conf.add_hline(y=70, line_dash="dash", line_color="#FF6B6B", line_width=2,
                           annotation_text="70% Threshold", annotation_position="top right",
                           annotation_font=dict(size=12, color='#FF6B6B', weight='bold'))

        st.plotly_chart(fig_conf, use_container_width=True)

        with st.expander("🔍 **View Detailed Analysis**", expanded=False):
            col_detail1, col_detail2 = st.columns(2)

            with col_detail1:
                st.markdown("#### 📏 **Measurement Analysis**")
                comparison = pd.DataFrame({
                    'Measurement': categories,
                    'Your Value': current_values,
                    'Species Average': [
                        avg_setosa[0] if predict_species == 'setosa' else avg_versicolor[
                            0] if predict_species == 'versicolor' else avg_virginica[0],
                        avg_setosa[1] if predict_species == 'setosa' else avg_versicolor[
                            1] if predict_species == 'versicolor' else avg_virginica[1],
                        avg_setosa[2] if predict_species == 'setosa' else avg_versicolor[
                            2] if predict_species == 'versicolor' else avg_virginica[2],
                        avg_setosa[3] if predict_species == 'setosa' else avg_versicolor[
                            3] if predict_species == 'versicolor' else avg_virginica[3]
                    ],
                    'Difference': [
                        current_values[0] - (avg_setosa[0] if predict_species == 'setosa' else avg_versicolor[
                            0] if predict_species == 'versicolor' else avg_virginica[0]),
                        current_values[1] - (avg_setosa[1] if predict_species == 'setosa' else avg_versicolor[
                            1] if predict_species == 'versicolor' else avg_virginica[1]),
                        current_values[2] - (avg_setosa[2] if predict_species == 'setosa' else avg_versicolor[
                            2] if predict_species == 'versicolor' else avg_virginica[2]),
                        current_values[3] - (avg_setosa[3] if predict_species == 'setosa' else avg_versicolor[
                            3] if predict_species == 'versicolor' else avg_virginica[3])
                    ]
                })
                st.dataframe(comparison.style.format({
                    'Your Value': '{:.2f}',
                    'Species Average': '{:.2f}',
                    'Difference': '{:+.2f}'
                }).applymap(
                    lambda x: 'color: #27ae60' if isinstance(x, (int,
                                                                 float)) and x > 0 else 'color: #e74c3c' if isinstance(
                        x, (int, float)) and x < 0 else '',
                    subset=['Difference']
                ), use_container_width=True, hide_index=True)

            with col_detail2:
                st.markdown("#### 🧬 **Species Information**")
                species_data = {
                    'setosa': {'Size': 'Small', 'Petal Color': 'White/Violet', 'Habitat': 'Open areas',
                               'Difficulty': 'Easy'},
                    'versicolor': {'Size': 'Medium', 'Petal Color': 'Blue-Violet', 'Habitat': 'Wet meadows',
                                   'Difficulty': 'Moderate'},
                    'virginica': {'Size': 'Large', 'Petal Color': 'Purple/Blue', 'Habitat': 'Coastal plains',
                                  'Difficulty': 'Easy'}
                }
                info = species_data[predict_species]
                for key, value in info.items():
                    st.markdown(f"""
                    <div class="feature-card">
                        <strong>{key}:</strong> {value}
                    </div>
                    """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 🌺 **Iris Species Guide**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 3rem;">🌸</div>
            <h3>Setosa</h3>
            <p><strong>Characteristics:</strong><br>
            • Smallest species<br>
            • Narrow petals<br>
            • Early spring bloom<br>
            • Open habitat preference</p>
            <div style="background: linear-gradient(135deg, #4ECDC4 0%, #44b8ad 100%); 
                        color: white; padding: 0.3rem; border-radius: 20px; margin-top: 0.5rem;">
                Easy to identify
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 3rem;">💜</div>
            <h3>Versicolor</h3>
            <p><strong>Characteristics:</strong><br>
            • Medium size<br>
            • Blue-violet flowers<br>
            • Late spring bloom<br>
            • Wetland habitat</p>
            <div style="background: linear-gradient(135deg, #45B7D1 0%, #3a9db8 100%); 
                        color: white; padding: 0.3rem; border-radius: 20px; margin-top: 0.5rem;">
                Moderate to identify
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 3rem;">🌼</div>
            <h3>Virginica</h3>
            <p><strong>Characteristics:</strong><br>
            • Largest species<br>
            • Wide petals<br>
            • Summer bloom<br>
            • Coastal preference</p>
            <div style="background: linear-gradient(135deg, #96CEB4 0%, #7db89a 100%); 
                        color: white; padding: 0.3rem; border-radius: 20px; margin-top: 0.5rem;">
                Distinctive features
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 **Species Measurement Comparison**")

    fig_box = go.Figure()

    for species in target_names:
        species_data = df[df['species_name'] == species]
        fig_box.add_trace(go.Box(
            y=species_data['petal length (cm)'],
            name=species.title(),
            boxmean='sd',  # Fixed: changed from 'boxmeanvisible' to 'boxmean'
            marker_color={'setosa': '#4ECDC4', 'versicolor': '#45B7D1', 'virginica': '#96CEB4'}[species],
            hovertemplate='<b>%{x}</b><br>Petal Length: %{y:.2f} cm<extra></extra>'
        ))

    fig_box.update_layout(
        title=dict(text="Petal Length Distribution by Species", font=dict(size=18, color='#1a202c', weight='bold')),
        yaxis_title=dict(text="Petal Length (cm)", font=dict(size=14, color='#1a202c', weight='bold')),
        xaxis_title=dict(text="Species", font=dict(size=14, color='#1a202c', weight='bold')),
        height=500,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif", size=12, color='#1a202c'),
        xaxis=dict(tickfont=dict(size=12, color='#1a202c', weight='bold')),
        yaxis=dict(tickfont=dict(size=12, color='#1a202c', weight='bold'), gridcolor='rgba(0,0,0,0.1)')
    )

    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.markdown("### 📈 **Dataset Statistics**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔗 **Feature Correlations**")
        corr = df[feature_names].corr()

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.str.replace(' (cm)', ''),
            y=corr.columns.str.replace(' (cm)', ''),
            colorscale='Viridis',
            text=corr.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12, "color": "white", "weight": "bold"},
            hoverongaps=False,
            hovertemplate='<b>%{x}</b> × <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))

        fig_corr.update_layout(
            height=400,
            title=dict(text="Correlation Matrix", font=dict(size=16, color='#1a202c', weight='bold')),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12, color='#1a202c'),
            xaxis=dict(tickfont=dict(size=11, color='#1a202c', weight='bold')),
            yaxis=dict(tickfont=dict(size=11, color='#1a202c', weight='bold'))
        )

        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        st.markdown("#### 🎯 **Species Distribution**")
        species_counts = df['species_name'].value_counts()

        fig_pie = go.Figure(data=[go.Pie(
            labels=species_counts.index.str.title(),
            values=species_counts.values,
            hole=0.3,
            marker_colors=['#4ECDC4', '#45B7D1', '#96CEB4'],
            textinfo='label+percent',
            textfont=dict(size=14, color='white', weight='bold'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])

        fig_pie.update_layout(
            height=400,
            title=dict(text="Dataset Balance", font=dict(size=16, color='#1a202c', weight='bold')),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12, color='#1a202c'),
            legend=dict(
                x=0.8,
                y=0.9,
                bgcolor='white',
                font=dict(size=12, color='#1a202c', weight='bold')
            )
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### 📊 **Summary Statistics**")
    st.dataframe(df[feature_names].describe().round(2), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: #f7fafc; border-radius: 20px;'>
    <p style='font-size: 1rem; margin: 0; color: #2d3748;'>🌸 <strong>Iris Classifier Pro</strong> | Powered by Random Forest (200 trees) | 95%+ Accuracy 🌸</p>
    <p style='font-size: 0.8rem; margin-top: 0.5rem; color: #4a5568;'>🎯 Real-time predictions | 📊 Interactive visualizations | 💡 Smart recommendations</p>
    <p style='font-size: 0.8rem; margin-top: 0.5rem; color: #4a5568;'>⚡ Tip: Adjust sliders and watch the radar chart update in real-time!</p>
</div>
""", unsafe_allow_html=True)