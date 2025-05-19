import os
import json
import pandas as pd
from argparse import ArgumentParser

def collect_genai_perf_data(root_dir, data_field):
    """
    Recursively collects performance data and returns a formatted DataFrame.
    
    Args:
        root_dir: Directory to search for JSON files
        data_field: Top-level field to extract (e.g., "request_throughput")
    
    Returns:
        Pivot DataFrame with cleaned artifact directories and sorted concurrency columns
    """
    data_records = []
    
    for root, _, files in os.walk(root_dir):
        if 'profile_export_genai_perf.json' in files:
            file_path = os.path.join(root, 'profile_export_genai_perf.json')
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Extract values with safety checks
                    raw_artifact = data.get('input_config', {}).get('output', {}).get('artifact_directory', '')
                    artifact = raw_artifact.replace('artifacts_', '', 1)  # Remove prefix
                    concurrency = data.get('input_config', {}).get('perf_analyzer', {}).get('stimulus', {}).get('concurrency')
                    value = data.get(data_field, {}).get('avg')

                    if data_field == "request_goodput":
                        value = value / data.get("request_throughput").get('avg')
                    
                    if all(v is not None for v in [artifact, concurrency, value]):
                        data_records.append({
                            'artifact': artifact,
                            'concurrency': concurrency,
                            'value': value
                        })
                    else:
                        print(f"Skipping {file_path} - missing required fields")
                        
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

    if data_records:
        df = pd.DataFrame(data_records)
        pivot_df = df.pivot_table(
            index='artifact',
            columns='concurrency',
            values='value'
        )
        # Sort columns by concurrency and clean format
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
        pivot_df.columns = [f'Concurrency {c}' for c in pivot_df.columns]
        return pivot_df.reset_index()
    return pd.DataFrame()

def print_formatted_table(df, data_field):
    """Prints a formatted table with title and cleaned headers"""
    if not df.empty:
        print(f"\n{data_field.replace('_', ' ').title()} Performance")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("No valid data found")

if __name__ == "__main__":
    parser = ArgumentParser(description='Analyze GenAI Performance Metrics')
    parser.add_argument('--directory', help='Root directory to search for JSON files',default='.')
    parser.add_argument('--metric',default="request_goodput", 
                      choices=['request_throughput', 'request_goodput',
                               'request_latency', 'time_to_first_token',
                               'output_token_throughput',"inter_token_latency",
                               "output_token_throughput_per_user"],
                      help='Metric to analyze')
    
    args = parser.parse_args()
    
    result_df = collect_genai_perf_data(args.directory, args.metric)
    print_formatted_table(result_df, args.metric)
