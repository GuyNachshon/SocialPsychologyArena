#!/usr/bin/env python3
"""Read and display the full conversation from experiment results."""

import pandas as pd
import sys
from pathlib import Path

def read_conversation(results_dir):
    """Read and display the full conversation."""
    conversation_file = Path(results_dir) / "conversation.parquet"
    
    if not conversation_file.exists():
        print(f"Conversation file not found: {conversation_file}")
        return
    
    # Read the conversation
    df = pd.read_parquet(conversation_file)
    
    print("=" * 80)
    print("FULL CONVERSATION LOG")
    print("=" * 80)
    print(f"Total messages: {len(df)}")
    print(f"Turns: {df['turn'].min()} to {df['turn'].max()}")
    print("=" * 80)
    
    # Display each message
    for _, row in df.iterrows():
        print(f"\n[Turn {row['turn']}] {row['speaker']} ({row['role']}):")
        print(f"  {row['message']}")
        if 'action_type' in row and pd.notna(row['action_type']):
            print(f"  [ACTION: {row['action_type']}]")
        print("-" * 60)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("CONVERSATION SUMMARY")
    print("=" * 80)
    
    # Messages per role
    role_counts = df['role'].value_counts()
    print("\nMessages per role:")
    for role, count in role_counts.items():
        print(f"  {role}: {count}")
    
    # Action types
    if 'action_type' in df.columns:
        action_counts = df['action_type'].value_counts()
        if len(action_counts) > 0:
            print("\nActions performed:")
            for action, count in action_counts.items():
                if pd.notna(action):
                    print(f"  {action}: {count}")
    
    # Timeline
    print(f"\nTimeline: {df['timestamp'].min()} to {df['timestamp'].max()}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_conversation.py <results_directory>")
        print("Example: python read_conversation.py 'logs/Stanford Prison Experiment_20250802_194321'")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    read_conversation(results_dir) 