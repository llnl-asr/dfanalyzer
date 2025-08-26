#!/usr/bin/env python3
"""
Command-line utility to summarize dftracer .pfw.gz traces.
Summaries are per-node, per-process, per-thread, or individual events.
Use --all-events with --node, --process, or --thread to output event-level summaries instead of grouped categories.
Add --aggregate to highlight the node with the max of each metric across nodes.
Use --level N to show call context up to depth N.

Usage examples:
  ./dfreport.py --node /path/to/COMPACT/
  ./dfreport.py --node --all-events /path/to/COMPACT/
  ./dfreport.py --process /path/to/COMPACT/
  ./dfreport.py --process --all-events /path/to/COMPACT/
  ./dfreport.py --thread /path/to/COMPACT/
  ./dfreport.py --thread --all-events /path/to/COMPACT/
  ./dfreport.py --events /path/to/COMPACT/
  ./dfreport.py --node --aggregate /path/to/COMPACT/
  ./dfreport.py --callctx --level 3 /path/to/COMPACT/

Options are mutually exclusive, one of --node, --process, --thread, --events, or --callctx is required.
"""
import argparse
import os
import glob
import gzip
import json
from collections import defaultdict
import pandas as pd
import dask.dataframe as dd
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for terminal colors
colorama_init()

#--------------- Load & Summaries ----------------

def assign_group(name: str) -> str:
    nl = name.lower()
    if nl.startswith('torchframework'):     return 'TorchFramework'
    if nl.startswith('pytorchdataloader'):  return 'PytorchDataLoader'
    if nl.startswith('pytorchcheckpointing'):return 'PytorchCheckpointing'
    if nl.startswith('filestorage'):        return 'FileStorage'
    if nl.startswith('dlio'):               return 'DLIO'
    if any(tok in nl for tok in ('open','close','start')): return 'file_ops'
    if any(tok in nl for tok in ('read','seek')):          return 'read_seek'
    if 'loop' in nl:                  return 'loop'
    if 'stat' in nl or 'xstat' in nl: return 'attr_checks'
    if 'npz' in nl:                   return 'npz_ops'
    return 'other'


def load_node_df(node_dir: str) -> pd.DataFrame|None:
    records=[]
    # Look for both .pfw.gz and .pfw files
    gz_files = glob.glob(os.path.join(node_dir, '*.pfw.gz'))
    pfw_files = glob.glob(os.path.join(node_dir, '*.pfw'))
    files = gz_files + pfw_files
    
    compact=os.path.join(node_dir,'COMPACT')
    if os.path.isdir(compact):
        files += glob.glob(os.path.join(compact, '*.pfw.gz'))
        files += glob.glob(os.path.join(compact, '*.pfw'))
        
    for p in sorted(files):
        if p.endswith('.pfw.gz'):
            # For gzipped files
            with gzip.open(p, 'rt') as f:
                for raw in f:
                    line = raw.strip().rstrip(',')
                    if not line or line in ('[',']'): continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records.append(obj)
        else:
            # For non-gzipped .pfw files
            with open(p, 'r') as f:
                for raw in f:
                    line = raw.strip().rstrip(',')
                    if not line or line in ('[',']'): continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records.append(obj)
                    
    if not records: return None
    
    # Create a pandas DataFrame
    df = pd.json_normalize(records)
    
    # Extract basic fields
    df['name'] = df.get('name', '').astype(str)
    df['dur'] = pd.to_numeric(df.get('dur', 0), errors='coerce').fillna(0.0)
    df['pid'] = pd.to_numeric(df.get('pid', 0), errors='coerce').fillna(0).astype(int)
    df['tid'] = pd.to_numeric(df.get('tid', 0), errors='coerce').fillna(0).astype(int)
    df['id'] = pd.to_numeric(df.get('id', 0), errors='coerce').fillna(0).astype(int)
    df['ts'] = pd.to_numeric(df.get('ts', 0), errors='coerce').fillna(0)
    df['ph'] = df.get('ph', '')
    
    # Calculate end time for duration events
    df['end'] = df['ts'] + df['dur']
    
    # Extract p_idx and level from args if available
    if 'args.p_idx' in df.columns:
        df['p_idx'] = pd.to_numeric(df['args.p_idx'], errors='coerce').fillna(-1).astype(int)
    elif 'args.p_idx' in df.columns:
        df['p_idx'] = pd.to_numeric(df['args.p_idx'], errors='coerce').fillna(-1).astype(int)
    else:
        # Try to extract from args dictionary
        if 'args' in df.columns:
            def extract_p_idx(args):
                if isinstance(args, dict) and 'p_idx' in args:
                    return args['p_idx']
                return -1
            df['p_idx'] = df['args'].apply(extract_p_idx)
            df['p_idx'] = pd.to_numeric(df['p_idx'], errors='coerce').fillna(-1).astype(int)
        else:
            df['p_idx'] = -1  # Default when no parent info available
    
    # Extract level from args
    if 'args.level' in df.columns:
        df['level'] = pd.to_numeric(df['args.level'], errors='coerce').fillna(0).astype(int)
    elif 'args' in df.columns:
        def extract_level(args):
            if isinstance(args, dict) and 'level' in args:
                return args['level']
            return 0
        df['level'] = df['args'].apply(extract_level)
        df['level'] = pd.to_numeric(df['level'], errors='coerce').fillna(0).astype(int)
    else:
        df['level'] = 0  # Default level
    
    # Basic required fields for summary
    basic_fields = ['name', 'dur', 'pid', 'tid']
    
    # Additional fields for call context analysis
    context_fields = ['id', 'ts', 'end', 'ph', 'p_idx', 'level']
    
    # Return all fields needed
    all_fields = basic_fields + context_fields
    return df[all_fields]


def build_call_context(df):
    """
    Build the calling context tree for each thread.
    
    Args:
        df: DataFrame containing trace events
    
    Returns:
        Dict mapping (pid, tid) to thread forest data
    """
    # Filter only duration events (ph == "X")
    df_duration = df[df['ph'] == 'X'].copy()
    
    # Group events by process and thread
    thread_groups = df_duration.groupby(['pid', 'tid'])
    
    # Store the forest for each thread
    thread_forests = {}
    
    for (pid, tid), thread_df in thread_groups:
        # Sort events by timestamp, end time, and id for consistent ordering
        thread_df = thread_df.sort_values(['ts', 'end', 'id'])
        
        # Build index by id for O(1) parent lookup
        ev_by_id = {}
        for idx, event in thread_df.iterrows():
            ev_by_id[event['id']] = {
                'event': event,
                'idx': idx,
                'node': None  # Will be populated as we build the tree
            }
        
        # Initialize forest roots for this thread
        roots = []
        
        # Track most recent node at each level for fallback parent detection
        level_stack = []
        
        # Track visited nodes to prevent cycles
        visited = set()
        
        # Process each event to build the tree
        for idx, event in thread_df.iterrows():
            # Skip if we've already processed this node
            if idx in visited:
                continue
                
            # Create node structure
            node = {
                'id': event['id'],
                'name': event['name'],
                'ts': event['ts'],
                'end': event['end'],
                'dur': event['dur'],
                'level': event['level'],
                'p_idx': event['p_idx'],
                'children': [],
                'parent': None
            }
            
            # Store node in event lookup
            ev_by_id[event['id']]['node'] = node
            visited.add(idx)
            
            # Try to attach to parent using p_idx
            if event['p_idx'] >= 0 and event['p_idx'] in ev_by_id:
                parent_id = event['p_idx']
                parent_data = ev_by_id[parent_id]
                
                # If parent node already created
                if parent_data['node'] is not None:
                    parent_node = parent_data['node']
                    
                    # Prevent cycles
                    parent_chain = []
                    temp_parent = parent_node
                    cycle_detected = False
                    
                    while temp_parent is not None:
                        if temp_parent['id'] == node['id']:
                            print(f"Warning: Cycle detected for event {node['id']}, breaking link")
                            cycle_detected = True
                            break
                        parent_chain.append(temp_parent['id'])
                        temp_parent = temp_parent['parent']
                    
                    if not cycle_detected:
                        # Link to parent
                        node['parent'] = parent_node
                        parent_node['children'].append(node)
                        
                        # Level sanity check (soft warning)
                        if node['level'] != parent_node['level'] + 1:
                            print(f"Notice: Level inconsistency - Event {node['id']} level {node['level']} " +
                                  f"but parent {parent_node['id']} level {parent_node['level']}")
                        
                        # Time containment check (soft warning)
                        if not (parent_node['ts'] <= node['ts'] and node['end'] <= parent_node['end']):
                            print(f"Notice: Time inconsistency - Event {node['id']} ({node['ts']}-{node['end']}) " +
                                  f"not contained in parent {parent_node['id']} ({parent_node['ts']}-{parent_node['end']})")
                    else:
                        # Add to roots if cycle detected
                        roots.append(node)
                else:
                    # This shouldn't happen with proper sorting, but handle it
                    print(f"Warning: Parent event {event['p_idx']} exists but node not created yet for event {event['id']}")
                    roots.append(node)
            else:
                # Fallback: use level stack (most recent node at each level)
                
                # Pop stack while current level <= node level (can't be parent)
                while level_stack and level_stack[-1]['level'] >= node['level']:
                    level_stack.pop()
                
                if level_stack:
                    # Last node in stack with level < current level is potential parent
                    potential_parent = level_stack[-1]
                    
                    # Check time containment as sanity check
                    if potential_parent['ts'] <= node['ts'] and node['end'] <= potential_parent['end']:
                        node['parent'] = potential_parent
                        potential_parent['children'].append(node)
                    else:
                        # Time inconsistency, add to roots
                        roots.append(node)
                else:
                    # No potential parent found, add to roots
                    roots.append(node)
                
                # Update level stack
                level_stack.append(node)
        
        # Store the forest for this thread
        thread_forests[(pid, tid)] = {
            'roots': roots,
            'events': ev_by_id
        }
    
    return thread_forests


def calculate_tree_metrics(node):
    """
    Calculate inclusive and exclusive time for a node and its children.
    
    Args:
        node: Tree node to process
        
    Returns:
        Tuple of (inclusive_time, exclusive_time)
    """
    # Inclusive time is the node's duration
    inclusive_time = node['dur']
    
    # Exclusive time starts as inclusive time, will subtract children's time
    exclusive_time = inclusive_time
    
    # Process children
    for child in node['children']:
        # Recursively calculate metrics for child
        child_inclusive, child_exclusive = calculate_tree_metrics(child)
        
        # Subtract child's inclusive time from parent's exclusive time
        # but cap at parent's duration to handle overlaps or timing issues
        child_time_capped = min(child_inclusive, inclusive_time)
        exclusive_time -= child_time_capped
    
    # Store metrics in the node
    node['inclusive_time'] = inclusive_time
    node['exclusive_time'] = max(0, exclusive_time)  # Ensure non-negative
    
    return inclusive_time, exclusive_time


def get_full_path(node):
    """Get the full path from root to this node."""
    path = []
    current = node
    while current:
        path.insert(0, current)
        current = current['parent']
    return path


def print_call_tree(node, indent=0, max_depth=None):
    """
    Print a node and its children as a tree.
    
    Args:
        node: Tree node to print
        indent: Current indentation level
        max_depth: Maximum depth to print
    """
    if max_depth is not None and indent > max_depth:
        return
    
    # Print current node
    prefix = "  " * indent
    inc_time = node.get('inclusive_time', node['dur'])
    exc_time = node.get('exclusive_time', node['dur'])
    
    print(f"{prefix}├─ {node['name']} [id={node['id']}] (dur={node['dur']:.2f}, inc={inc_time:.2f}, exc={exc_time:.2f})")
    
    # Print children
    for child in sorted(node['children'], key=lambda x: x['inclusive_time'], reverse=True):
        print_call_tree(child, indent + 1, max_depth)


def print_call_context(thread_forests, level_depth=None):
    """
    Print the call context trees for all threads.
    
    Args:
        thread_forests: Dict mapping (pid, tid) to thread forest data
        level_depth: Maximum depth to print
    """
    # Calculate metrics for all trees
    for (pid, tid), forest in thread_forests.items():
        for root in forest['roots']:
            calculate_tree_metrics(root)
    
    # Print trees for each thread
    for (pid, tid), forest in sorted(thread_forests.items()):
        print(f"\n=== Call Context for Process {pid}, Thread {tid} ===")
        
        # Sort roots by inclusive time
        sorted_roots = sorted(forest['roots'], key=lambda x: x.get('inclusive_time', x['dur']), reverse=True)
        
        for root in sorted_roots:
            print_call_tree(root, 0, level_depth)


def aggregate_call_paths(thread_forests, max_level):
    """
    Aggregate call paths up to specified level.
    
    Args:
        thread_forests: Dict mapping (pid, tid) to thread forest data
        max_level: Maximum path depth to include
        
    Returns:
        DataFrame with aggregated metrics by call path
    """
    # Calculate metrics for all trees if not already done
    for (pid, tid), forest in thread_forests.items():
        for root in forest['roots']:
            if 'inclusive_time' not in root:
                calculate_tree_metrics(root)
    
    # Collect all paths
    all_paths = []
    
    # Helper function to traverse the tree and collect paths
    def collect_paths(node, current_path=None, depth=0):
        if current_path is None:
            current_path = []
        
        # Add current node to path
        path = current_path + [node]
        
        # If we're at or below max_level, record this path
        if depth <= max_level:
            # Create path signature for aggregation
            path_sig = " → ".join(n['name'] for n in path)
            
            all_paths.append({
                'path_signature': path_sig,
                'path': path,
                'depth': depth,
                'inclusive_time': node['inclusive_time'],
                'exclusive_time': node['exclusive_time'],
                'pid': pid,
                'tid': tid,
                'node_id': node['id'],
                'duration': node['dur']
            })
        
        # Continue traversal for children
        for child in node['children']:
            collect_paths(child, path, depth + 1)
    
    # Collect paths from all roots
    for (pid, tid), forest in thread_forests.items():
        for root in forest['roots']:
            collect_paths(root)
    
    # Convert to DataFrame for easier aggregation
    if not all_paths:
        return pd.DataFrame()
        
    paths_df = pd.DataFrame(all_paths)
    
    # Group by path signature
    aggregated = paths_df.groupby('path_signature').agg(
        count=('node_id', 'count'),
        total_inclusive=('inclusive_time', 'sum'),
        avg_inclusive=('inclusive_time', 'mean'),
        total_exclusive=('exclusive_time', 'sum'),
        avg_exclusive=('exclusive_time', 'mean'),
        max_inclusive=('inclusive_time', 'max'),
        min_inclusive=('inclusive_time', 'min'),
        std_inclusive=('inclusive_time', 'std')
    ).reset_index()
    
    # Sort by total inclusive time
    aggregated = aggregated.sort_values('total_inclusive', ascending=False)
    
    return aggregated


def summarize_groups(df):
    df = df[df['dur']>0].copy()
    df['group'] = df['name'].apply(assign_group)
    
    # Convert to Dask only for larger dataframes
    if len(df) > 10000:  # Threshold for using Dask
        ddf = dd.from_pandas(df, npartitions=max(1, len(df) // 10000))
        total = ddf['dur'].sum().compute()
        
        # Use Dask aggregation
        agg_df = ddf.groupby('group')['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index().compute()
    else:
        total = df['dur'].sum()
        # Use pandas for smaller dataframes
        agg_df = df.groupby('group')['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index()
    
    agg_df['% Total Time'] = (100 * agg_df['Total_Time'] / total).round(3)
    for c in ['Total_Time','Average','Min','Max','StdDev']:
        agg_df[c] = agg_df[c].round(3)
    
    return agg_df.sort_values('% Total Time', ascending=False)[['group','% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']]


def summarize_events(df):
    df = df[df['dur']>0]
    
    # Convert to Dask only for larger dataframes
    if len(df) > 10000:  # Threshold for using Dask
        ddf = dd.from_pandas(df, npartitions=max(1, len(df) // 10000))
        total = ddf['dur'].sum().compute()
        
        # Use Dask aggregation
        agg_df = ddf.groupby('name')['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index().compute()
    else:
        total = df['dur'].sum()
        # Use pandas for smaller dataframes
        agg_df = df.groupby('name')['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index()
    
    agg_df['% Total Time'] = (100 * agg_df['Total_Time'] / total).round(3)
    for c in ['Total_Time','Average','Min','Max','StdDev']:
        agg_df[c] = agg_df[c].round(3)
    
    return agg_df.sort_values('% Total Time', ascending=False)[['name','% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']]


def summarize_process_groups(df):
    df = df[df['dur']>0].copy()
    df['group'] = df['name'].apply(assign_group)
    
    # Convert to Dask only for larger dataframes
    if len(df) > 10000:  # Threshold for using Dask
        ddf = dd.from_pandas(df, npartitions=max(1, len(df) // 10000))
        
        # Use Dask aggregation
        agg_df = ddf.groupby(['pid','group'])['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index().compute()
        
        # Calculate pid_total using Dask
        pct_df = ddf.groupby('pid')['dur'].sum().reset_index().rename(columns={'dur': 'pid_total'}).compute()
    else:
        # Use pandas for smaller dataframes
        agg_df = df.groupby(['pid','group'])['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index()
        
        # Calculate pid_total using pandas
        pct_df = df.groupby('pid')['dur'].sum().reset_index().rename(columns={'dur': 'pid_total'})
    
    # Merge results
    agg_df = agg_df.merge(pct_df, on='pid')
    agg_df['% Total Time'] = (100 * agg_df['Total_Time'] / agg_df['pid_total']).round(3)
    
    for c in ['Total_Time','Average','Min','Max','StdDev']:
        agg_df[c] = agg_df[c].round(3)
    
    return agg_df.sort_values(['pid','% Total Time'], ascending=[True,False])[['pid','group','% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']]


def summarize_thread_groups(df):
    df = df[df['dur']>0].copy()
    df['group'] = df['name'].apply(assign_group)
    
    # Convert to Dask only for larger dataframes
    if len(df) > 10000:  # Threshold for using Dask
        ddf = dd.from_pandas(df, npartitions=max(1, len(df) // 10000))
        
        # Use Dask aggregation
        agg_df = ddf.groupby(['tid','group'])['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index().compute()
        
        # Calculate tid_total using Dask
        pct_df = ddf.groupby('tid')['dur'].sum().reset_index().rename(columns={'dur': 'tid_total'}).compute()
    else:
        # Use pandas for smaller dataframes
        agg_df = df.groupby(['tid','group'])['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index()
        
        # Calculate tid_total using pandas
        pct_df = df.groupby('tid')['dur'].sum().reset_index().rename(columns={'dur': 'tid_total'})
    
    # Merge results
    agg_df = agg_df.merge(pct_df, on='tid')
    agg_df['% Total Time'] = (100 * agg_df['Total_Time'] / agg_df['tid_total']).round(3)
    
    for c in ['Total_Time','Average','Min','Max','StdDev']:
        agg_df[c] = agg_df[c].round(3)
    
    return agg_df.sort_values(['tid','% Total Time'], ascending=[True,False])[['tid','group','% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']]


def build_group_map(df):
    # Use pandas directly for better compatibility
    unique_names = df['name'].unique()
    
    gm = defaultdict(set)
    for nm in unique_names:
        gm[assign_group(nm)].add(nm)
    
    return {g: sorted(list(ev)) for g, ev in gm.items()}


def print_tree_for_node(node, df, group_map, mode, all_ev):
    print(f"\n===== Summary for {node} =====\n")
    if mode=='events':
        print(summarize_events(df).to_string(index=False)); return
    if mode=='node':
        print((summarize_events(df) if all_ev else summarize_groups(df)).to_string(index=False));return
    if mode=='process':
        if all_ev:
            for pid,sub in df.groupby('pid'):
                print(f"--- Process {pid} ---")
                print(summarize_events(sub).to_string(index=False));print("============")
        else:
            for pid,sub in summarize_process_groups(df).groupby('pid'):
                print(f"--- Process {pid} ---")
                print(sub.drop(columns='pid').to_string(index=False));print("============")
        return
    if mode=='thread':
        if all_ev:
            for tid,sub in df.groupby('tid'):
                print(f"--- Thread {tid} ---")
                print(summarize_events(sub).to_string(index=False));print("-------")
        else:
            for tid,sub in summarize_thread_groups(df).groupby('tid'):
                print(f"--- Thread {tid} ---")
                print(sub.drop(columns='tid').to_string(index=False));print("-------")
        return
    if mode=='callctx':
        # Build and print call context trees
        forests = build_call_context(df)
        print_call_context(forests, level_depth=args.level if 'args' in locals() else None)
        return


def highlight_across_nodes(per_node, key_col, metrics):
    nodes = list(per_node.keys())
    
    # Extract unique keys from all DataFrames
    keys_set = set()
    for df in per_node.values():
        keys_set.update(df[key_col].tolist())
    keys = sorted(keys_set)
    
    for key in keys:
        print(f"\n>>> {key}")
        
        # Find max holder for each metric using improved approach
        max_holder = {}
        for m in metrics:
            max_val = float('-inf')
            max_node = None
            for n in nodes:
                row = per_node[n]
                if key in row[key_col].values:
                    val = row.set_index(key_col).get(m, 0).get(key, 0)
                    if val > max_val:
                        max_val = val
                        max_node = n
            max_holder[m] = max_node
        
        hdr = "    node" + "".join(m.rjust(12) for m in metrics)
        print(hdr)
        
        for n in nodes:
            row = per_node[n]
            if key in row[key_col].values:
                vals = {m: row.set_index(key_col).get(m, 0).get(key, 0) for m in metrics}
                line = "    " + n.ljust(11)
                for m in metrics:
                    col = Fore.RED if n == max_holder[m] else Fore.GREEN
                    line += col + f"{vals[m]:12.3f}" + Style.RESET_ALL
                print(line)
            else:
                # Print a placeholder for nodes that don't have this key
                line = "    " + n.ljust(11) + "".join(" " * 12 for _ in metrics)
                print(line)


def print_call_path_summary(df, node_name):
    """Print aggregated call path summary."""
    if df.empty:
        print(f"No call paths found for {node_name}")
        return
        
    # Format the output
    print(f"\n===== Call Path Summary for {node_name} =====\n")
    
    # Display top paths by total inclusive time
    print(f"Top Call Paths (by total inclusive time):")
    
    # Format the DataFrame for display
    display_df = df.copy()
    display_df['total_inclusive'] = display_df['total_inclusive'].round(3)
    display_df['avg_inclusive'] = display_df['avg_inclusive'].round(3)
    display_df['total_exclusive'] = display_df['total_exclusive'].round(3)
    display_df['avg_exclusive'] = display_df['avg_exclusive'].round(3)
    
    # Rename columns for better readability
    display_df = display_df.rename(columns={
        'path_signature': 'Call Path',
        'count': 'Count',
        'total_inclusive': 'Total Inc Time',
        'avg_inclusive': 'Avg Inc Time',
        'total_exclusive': 'Total Exc Time',
        'avg_exclusive': 'Avg Exc Time'
    })
    
    # Display subset of columns
    columns_to_display = ['Call Path', 'Count', 'Total Inc Time', 'Avg Inc Time', 'Total Exc Time', 'Avg Exc Time']
    print(display_df[columns_to_display].to_string(index=False))


def main():
    p = argparse.ArgumentParser(description='Analyze dftracer traces with call context support')
    p.add_argument('directory', help='Directory containing trace files')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--node', action='store_true', help='Summarize by node')
    g.add_argument('--process', action='store_true', help='Summarize by process')
    g.add_argument('--thread', action='store_true', help='Summarize by thread')
    g.add_argument('--events', action='store_true', help='List all events')
    g.add_argument('--callctx', action='store_true', help='Analyze call context')
    p.add_argument('--all-events', action='store_true', help='Show all events instead of group summaries')
    p.add_argument('--aggregate', action='store_true', help='Highlight the node with the max of each metric across nodes')
    p.add_argument('--level', type=int, default=0, help='Maximum depth level to display in call context or call path')
    args = p.parse_args()
    
    base = args.directory
    if not os.path.isdir(base): 
        p.error(f"{base} not a directory")
    
    # Check for both .pfw.gz and .pfw files
    has_pfw_files = glob.glob(os.path.join(base,'*.pfw.gz')) or glob.glob(os.path.join(base,'*.pfw'))
    if has_pfw_files: 
        nodes = [base]
    else: 
        nodes = [os.path.join(base,d) for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base,d))]
    
    raw = {}
    for nd in nodes:
        df = load_node_df(nd)
        if df is None: continue
        nm = os.path.basename(nd.rstrip(os.sep))
        raw[nm if nm.lower()!='compact' else os.path.basename(os.path.dirname(nd))] = df
    
    if not raw:
        print("No traces found"); return
        
    # Use pandas concat for better compatibility
    combined_df = pd.concat(raw.values(), ignore_index=True)
    group_map = build_group_map(combined_df)
        
    mode = ('callctx' if args.callctx else 
            'events' if args.events else 
            'process' if args.process else 
            'thread' if args.thread else 
            'node')
    
    # Special handling for call context mode
    if mode == 'callctx':
        if args.aggregate:
            # Aggregate call paths across nodes
            all_path_dfs = {}
            for n, df in raw.items():
                forests = build_call_context(df)
                path_df = aggregate_call_paths(forests, args.level)
                all_path_dfs[n] = path_df
                print_call_path_summary(path_df, n)
            
            # TODO: Add cross-node path comparison if needed
        else:
            # Process each node individually
            for n, df in raw.items():
                forests = build_call_context(df)
                if args.level > 0:
                    # Show aggregated call paths
                    path_df = aggregate_call_paths(forests, args.level)
                    print_call_path_summary(path_df, n)
                else:
                    # Show full call trees
                    print_call_context(forests)
        return
    
    # Standard processing for other modes
    for n, df in raw.items():
        print_tree_for_node(n, df, group_map, mode, args.all_events)

    # Global aggregate highlighting for other modes
    if args.aggregate and not (mode == 'process'):
        summ = {}
        metrics = ['% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']
        for n, df in raw.items():
            if mode=='node':
                summ[n] = summarize_events(df) if args.all_events else summarize_groups(df)
            elif mode=='events':
                summ[n] = summarize_events(df)
            elif mode=='process':
                summ[n] = summarize_events(df) if args.all_events else summarize_process_groups(df)
            else:
                summ[n] = summarize_events(df) if args.all_events else summarize_thread_groups(df)
        key_col = ('name' if mode=='events' else 'group')
        highlight_across_nodes(summ, key_col, metrics)

if __name__ == '__main__': 
    main()

