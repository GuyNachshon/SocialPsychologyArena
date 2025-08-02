"""Streamlit dashboard for Social Psych Arena results."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from psy_lab.metrics import MetricEngine


def load_experiment_data(results_dir: str):
    """Load experiment data from results directory."""
    results_path = Path(results_dir)
    
    # Load conversation data
    conversation_file = results_path / "conversation.parquet"
    if conversation_file.exists():
        conversation_df = pd.read_parquet(conversation_file)
    else:
        conversation_df = pd.DataFrame()
    
    # Load private messages data
    private_messages_file = results_path / "private_messages.parquet"
    if private_messages_file.exists():
        private_messages_df = pd.read_parquet(private_messages_file)
    else:
        private_messages_df = pd.DataFrame()
    
    # Load metrics data
    metrics_file = results_path / "metrics.parquet"
    if metrics_file.exists():
        metrics_df = pd.read_parquet(metrics_file)
    else:
        metrics_df = pd.DataFrame()
    
    # Load metadata
    metadata_file = results_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return conversation_df, private_messages_df, metrics_df, metadata


def plot_conversation_timeline(conversation_df: pd.DataFrame):
    """Plot conversation timeline."""
    if conversation_df.empty:
        st.warning("No conversation data available")
        return
    
    fig = px.scatter(
        conversation_df,
        x='turn',
        y='speaker',
        color='role',
        size=[20] * len(conversation_df),  # Fixed size for all points
        hover_data=['message'],
        title="Conversation Timeline by Speaker",
        labels={'turn': 'Turn Number', 'speaker': 'Speaker', 'role': 'Role'}
    )
    
    fig.update_layout(
        xaxis_title="Turn Number",
        yaxis_title="Speaker",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_metrics_over_time(metrics_df: pd.DataFrame):
    """Plot metrics over time."""
    if metrics_df.empty:
        st.warning("No metrics data available")
        return
    
    # Create subplots for each metric
    metric_columns = [col for col in metrics_df.columns if col != 'turn']
    
    if len(metric_columns) == 0:
        st.warning("No metrics found in data")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=len(metric_columns),
        cols=1,
        subplot_titles=metric_columns,
        vertical_spacing=0.1
    )
    
    for i, metric in enumerate(metric_columns, 1):
        fig.add_trace(
            go.Scatter(
                x=metrics_df.index,
                y=metrics_df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(width=2)
            ),
            row=i, col=1
        )
    
    fig.update_layout(
        height=200 * len(metric_columns),
        title_text="Metrics Over Time",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_metric_distributions(metrics_df: pd.DataFrame):
    """Plot metric distributions."""
    if metrics_df.empty:
        st.warning("No metrics data available")
        return
    
    metric_columns = [col for col in metrics_df.columns if col != 'turn']
    
    if len(metric_columns) == 0:
        st.warning("No metrics found in data")
        return
    
    # Create violin plots
    fig = go.Figure()
    
    for metric in metric_columns:
        fig.add_trace(
            go.Violin(
                y=metrics_df[metric],
                name=metric,
                box_visible=True,
                meanline_visible=True
            )
        )
    
    fig.update_layout(
        title="Metric Distributions",
        yaxis_title="Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_conversation_excerpts(conversation_df: pd.DataFrame, n_excerpts: int = 5):
    """Display conversation excerpts."""
    if conversation_df.empty:
        st.warning("No conversation data available")
        return
    
    st.subheader("Recent Conversation Excerpts")
    
    # Get recent messages
    recent_messages = conversation_df.tail(n_excerpts)
    
    for _, message in recent_messages.iterrows():
        with st.expander(f"Turn {message['turn']}: {message['speaker']} ({message['role']})"):
            st.write(f"**Message:** {message['message']}")
            st.write(f"**Timestamp:** {message['timestamp']}")


def display_full_conversation_chat(conversation_df: pd.DataFrame):
    """Display the full conversation in a chat-style interface."""
    if conversation_df.empty:
        st.warning("No conversation data available")
        return
    
    st.subheader("Full Conversation (Chat Style)")
    
    # Add summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Messages", len(filtered_df))
    with col2:
        st.metric("Turns", filtered_df['turn'].max() if not filtered_df.empty else 0)
    with col3:
        st.metric("Speakers", filtered_df['speaker'].nunique())
    with col4:
        st.metric("Actions", len(filtered_df[filtered_df.get('action_type', '').notna()]))
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_actions = st.checkbox("Show Actions", value=True)
    with col2:
        show_timestamps = st.checkbox("Show Timestamps", value=False)
    with col3:
        filter_role = st.selectbox("Filter by Role", ["All"] + list(conversation_df['role'].unique()))
    
    # Filter data
    filtered_df = conversation_df.copy()
    if filter_role != "All":
        filtered_df = filtered_df[filtered_df['role'] == filter_role]
    
    # Create chat container with scrolling
    st.markdown("""
    <style>
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        background-color: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        # Style for chat messages
        st.markdown("""
        <style>
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            border-left: 6px solid #ccc;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
        }
        .guard-message {
            background-color: #e8f4fd;
            border-left-color: #1976d2;
            color: #0d47a1;
        }
        .prisoner-message {
            background-color: #fff8e1;
            border-left-color: #f57c00;
            color: #e65100;
        }
        .warden-message {
            background-color: #f3e5f5;
            border-left-color: #7b1fa2;
            color: #4a148c;
        }
        .action-message {
            background-color: #e8f5e8;
            border-left-color: #388e3c;
            color: #1b5e20;
            font-weight: 500;
        }
        .chat-message strong {
            color: #2c3e50;
            font-size: 1.1em;
            display: block;
            margin-bottom: 0.5rem;
        }
        .chat-message p {
            margin: 0.5rem 0;
            color: inherit;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display messages
        for _, message in filtered_df.iterrows():
            # Determine message type and styling
            role = message['role']
            is_action = 'action_type' in message and pd.notna(message['action_type'])
            
            if is_action and show_actions:
                css_class = "action-message"
                icon = "⚡"
                prefix = f"**{icon} ACTION:** "
            else:
                if role == "guard":
                    css_class = "guard-message"
                    icon = "👮"
                elif role == "prisoner":
                    css_class = "prisoner-message"
                    icon = "👤"
                elif role == "warden":
                    css_class = "warden-message"
                    icon = "👑"
                else:
                    css_class = "chat-message"
                    icon = "💬"
                
                prefix = ""
            
            # Create message content
            message_content = f"{prefix}{message['message']}"
            
            # Add timestamp if requested
            if show_timestamps:
                timestamp = message['timestamp']
                message_content += f"\n\n*{timestamp}*"
            
            # Display message
            st.markdown(f"""
            <div class="chat-message {css_class}">
                <strong>{icon} Turn {message['turn']}: {message['speaker']} ({role})</strong>
                <p>{message_content}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add download button for conversation
    st.download_button(
        label="Download Full Conversation (CSV)",
        data=filtered_df.to_csv(index=False),
        file_name="full_conversation.csv",
        mime="text/csv"
    )


def display_conversation_statistics(conversation_df: pd.DataFrame):
    """Display conversation statistics."""
    if conversation_df.empty:
        st.warning("No conversation data available")
        return
    
    st.subheader("Conversation Statistics")
    
    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", len(conversation_df))
    
    with col2:
        st.metric("Total Turns", conversation_df['turn'].max())
    
    with col3:
        st.metric("Unique Speakers", conversation_df['speaker'].nunique())
    
    with col4:
        st.metric("Unique Roles", conversation_df['role'].nunique())
    
    # Messages per role
    st.subheader("Messages by Role")
    role_counts = conversation_df['role'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=role_counts.values,
            names=role_counts.index,
            title="Messages by Role"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(role_counts.reset_index().rename(columns={'index': 'Role', 'role': 'Count'}))
    
    # Actions performed
    if 'action_type' in conversation_df.columns:
        st.subheader("Actions Performed")
        action_counts = conversation_df['action_type'].value_counts()
        if len(action_counts) > 0:
            fig = px.bar(
                x=action_counts.index,
                y=action_counts.values,
                title="Actions by Type",
                labels={'x': 'Action Type', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No actions recorded in this conversation")
    
    # Turn distribution
    st.subheader("Message Distribution Over Turns")
    turn_counts = conversation_df['turn'].value_counts().sort_index()
    fig = px.line(
        x=turn_counts.index,
        y=turn_counts.values,
        title="Messages per Turn",
        labels={'x': 'Turn', 'y': 'Message Count'}
    )
    st.plotly_chart(fig, use_container_width=True)


def display_private_messages(private_messages_df: pd.DataFrame):
    """Display private messages between agents."""
    if private_messages_df.empty:
        st.info("No private messages recorded in this experiment")
        return
    
    st.subheader("Private Messages")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        filter_channel = st.selectbox("Filter by Channel", ["All"] + list(private_messages_df['channel'].unique()))
    with col2:
        filter_type = st.selectbox("Filter by Message Type", ["All"] + list(private_messages_df['message_type'].unique()))
    
    # Filter data
    filtered_df = private_messages_df.copy()
    if filter_channel != "All":
        filtered_df = filtered_df[filtered_df['channel'] == filter_channel]
    if filter_type != "All":
        filtered_df = filtered_df[filtered_df['message_type'] == filter_type]
    
    # Display private messages
    for _, message in filtered_df.iterrows():
        with st.expander(f"Turn {message['turn']}: {message['from']} → {message['to']} ({message['message_type']})"):
            st.write(f"**Message:** {message['message']}")
            st.write(f"**Channel:** {message['channel']}")
            st.write(f"**Timestamp:** {message['timestamp']}")
    
    # Private message statistics
    st.subheader("Private Message Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Messages per channel
        channel_counts = private_messages_df['channel'].value_counts()
        fig = px.pie(
            values=channel_counts.values,
            names=channel_counts.index,
            title="Private Messages by Channel"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Messages per type
        type_counts = private_messages_df['message_type'].value_counts()
        fig = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            title="Private Messages by Type",
            labels={'x': 'Message Type', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)


def display_metadata(metadata: dict):
    """Display experiment metadata."""
    if not metadata:
        st.warning("No metadata available")
        return
    
    st.subheader("Experiment Metadata")
    
    # Scenario info
    if 'scenario' in metadata:
        scenario = metadata['scenario']
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name:** {scenario.get('name', 'N/A')}")
            st.write(f"**Description:** {scenario.get('description', 'N/A')}")
            st.write(f"**Seed:** {scenario.get('seed', 'N/A')}")
        
        with col2:
            st.write(f"**Total Agents:** {scenario.get('roles', [])}")
            st.write(f"**Max Turns:** {scenario.get('stop_criteria', {}).get('max_turns', 'N/A')}")
            st.write(f"**Max Cost:** ${scenario.get('stop_criteria', {}).get('max_cost', 'N/A')}")
    
    # State info
    if 'state' in metadata:
        state = metadata['state']
        st.write(f"**Total Turns:** {state.get('turn', 'N/A')}")
        st.write(f"**Stop Reason:** {state.get('stop_reason', 'N/A')}")
        st.write(f"**Total Tokens:** {state.get('total_tokens', 'N/A')}")
        st.write(f"**Total Cost:** ${state.get('total_cost', 'N/A')}")


def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="Social Psych Arena Dashboard",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Social Psych Arena Dashboard")
    st.markdown("Visualize and analyze social psychology experiment results")
    
    # Get results directory from command line args or use default
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Try to find the most recent results directory
        logs_dir = Path("logs")
        if logs_dir.exists():
            results_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
            if results_dirs:
                results_dir = str(max(results_dirs, key=lambda x: x.stat().st_mtime))
            else:
                results_dir = ""
        else:
            results_dir = ""
    
    # File uploader for results
    uploaded_file = st.file_uploader(
        "Upload results directory (zip file) or use existing results",
        type=['zip']
    )
    
    if uploaded_file is not None:
        # Handle uploaded file
        st.info("File upload functionality not implemented yet")
        return
    
    # Load data
    if results_dir and Path(results_dir).exists():
        conversation_df, private_messages_df, metrics_df, metadata = load_experiment_data(results_dir)
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page",
            ["Overview", "Full Conversation", "Conversation Stats", "Private Messages", "Metrics", "Analysis"]
        )
        
        if page == "Overview":
            st.header("Experiment Overview")
            display_metadata(metadata)
            
            # Summary statistics
            if not metrics_df.empty:
                st.subheader("Metrics Summary")
                summary_stats = metrics_df.describe()
                st.dataframe(summary_stats)
        
        elif page == "Full Conversation":
            st.header("Full Conversation Analysis")
            display_full_conversation_chat(conversation_df)
        
        elif page == "Conversation Stats":
            st.header("Conversation Statistics")
            plot_conversation_timeline(conversation_df)
            display_conversation_statistics(conversation_df)
            display_conversation_excerpts(conversation_df)
        
        elif page == "Private Messages":
            st.header("Private Messages Analysis")
            display_private_messages(private_messages_df)
        
        elif page == "Metrics":
            st.header("Metrics Analysis")
            plot_metrics_over_time(metrics_df)
            plot_metric_distributions(metrics_df)
        
        elif page == "Analysis":
            st.header("Advanced Analysis")
            
            # Custom analysis options
            if not metrics_df.empty:
                st.subheader("Metric Correlations")
                correlation_matrix = metrics_df.corr()
                st.dataframe(correlation_matrix)
                
                # Correlation heatmap
                fig = px.imshow(
                    correlation_matrix,
                    title="Metric Correlations",
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No results directory found. Please run an experiment first or upload results.")
        st.info("To run an experiment, use: `psy-lab run scenarios/asch.yaml`")


if __name__ == "__main__":
    main() 