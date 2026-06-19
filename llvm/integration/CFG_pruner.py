from typing import Dict, Any, Tuple
import sys
import os
import pydot
from pydot import Dot
from pyvis.network import Network
import networkx as nx
import argparse
import re
from collections import defaultdict
from tqdm import tqdm

#!/usr/bin/env python3
"""
CFG_pruner.py - small helper to read a Graphviz DOT file into a NetworkX graph.

Usage:
    python CFG_pruner.py path/to/graph_files perf_trace_file [--total_time_limit 0.01] [--top_n 5] [--output cfgs_annotated] [--entry-points]

"""

def remove_empty_elements(g: Dot):
    for node in g.get_nodes():
        if node.get_name() == "" or node.get_label() == "" or node.get_label() is None:
            g.del_node(node.get_name())
    for edge in g.get_edges():
        if edge.get_source() == "" or edge.get_destination() == "" or edge.get_label() == "" or edge.get_label() is None:
            g.del_edge(edge.get_source(), edge.get_destination())

def read_dot_file(path: str):
    try:
        # Try using pygraphviz via networkx for faster parsing
        graphs = [nx.drawing.nx_pydot.to_pydot(nx.nx_agraph.read_dot(path))]
    except Exception:
        graphs = pydot.graph_from_dot_file(path)
    
    # graphs = pydot.graph_from_dot_file(path)
    graph = graphs[0]
    remove_empty_elements(graph)
    return graph

def _print_summary(g: Dot):
    print("Graph Summary:")
    print("Number of nodes:", len(g.get_nodes()))
    print("Number of edges:", len(g.get_edges()))
    print("root node:", g.get_node_list()[0].get_label())
    call_nodes = [node for node in g.get_nodes() if "()" in node.get_label()]
    print("Number of call nodes:", len(call_nodes))

def remove_node(g: Dot, node_name: str, in_edges, out_edges):
    node_to_remove = g.get_node(node_name)
    if node_to_remove:
        g.del_node(node_name)
        # Remove incoming edges
        for edge in in_edges.get(node_name, []):
            g.del_edge(edge.get_source(), edge.get_destination())
        # Remove outgoing edges
        for edge in out_edges.get(node_name, []):
            g.del_edge(edge.get_source(), edge.get_destination())

def bfs(g: Dot, in_edges, out_edges) -> set:
    root = g.get_node_list()[0]
    visited = set()
    queue = [root.get_name()]
    while queue:
        current = queue.pop(0)
        visited.add(current)
        for edge in in_edges.get(current, []):
            if edge.get_dir() != "both":
                continue
            src = edge.get_source()
            if src not in visited and src not in queue:
                queue.append(src)
        for edge in out_edges.get(current, []):
            dst = edge.get_destination()
            if dst not in visited and dst not in queue:
                queue.append(dst)
    return visited

def remove_orphaned_nodes(g: Dot):
    in_edges, out_edges = build_adjacency_index(g)
    if len(g.get_node_list()) < 1:
        return
    visited = bfs(g, in_edges, out_edges)
    print("Number of reachable nodes after pruning:", len(visited))
    for node in g.get_nodes():
        if node.get_name() not in visited:
            remove_node(g, node.get_name(), in_edges, out_edges)

def filter_prunable_nodes(g: Dot, total_time_limit: float, duration: float, children: defaultdict[str, set]) -> list:
    prunable_nodes = []
    call_nodes = [node for node in g.get_nodes() if "()" in node.get_label() and "main()" not in node.get_label()]

    root = g.get_node_list()[0]

    for call_node in call_nodes:
        if call_node.get_name() == root.get_name():
            continue
        total_time = totaltimes.get(call_node.get_name(), 0)
        if total_time < total_time_limit * duration:
            for child in children.get(call_node.get_name(), []):
                child_time = totaltimes.get(child, 0)
                if child_time >= total_time_limit * duration:
                    total_time = child_time
        if total_time < total_time_limit * duration:
                prunable_nodes.append(call_node)

    return prunable_nodes

def build_adjacency_index(graph):
    in_edges = defaultdict(list)
    out_edges = defaultdict(list)
    
    for edge in graph.get_edge_list():
        source = edge.get_source().split(":")[0]
        destination = edge.get_destination().split(":")[0]
        
        out_edges[source].append(edge)
        in_edges[destination].append(edge)
        
    return in_edges, out_edges

def replace_node(g: Dot, old_node_name: str, new_node_name: str, in_edges, out_edges):
    node_to_remove = g.get_node(old_node_name)
    if not node_to_remove:
        return
    node_to_remove = node_to_remove[0]
    label = node_to_remove.get_label()
    shape = node_to_remove.get_shape()
    style = node_to_remove.get_style()
    if style is None:
        style = "filled"
    fillcolor = node_to_remove.get_fillcolor()
    if fillcolor is None:
        fillcolor = "white"
    g.add_node(pydot.Node(new_node_name, label=label, shape=shape, style=style, fillcolor=fillcolor))
    if node_to_remove:
        g.del_node(old_node_name)
        # Remove outgoing edges
        for edge in out_edges.get(old_node_name, []):
            g.del_edge(edge.get_source(), edge.get_destination())

        # Replace incoming edges
        for edge in in_edges.get(old_node_name, []):
            g.add_edge(pydot.Edge(edge.get_source(), new_node_name, label=edge.get_label(), dir=edge.get_dir()))
            g.del_edge(edge.get_source(), edge.get_destination())

def prune_leafs(g: Dot, total_time_limit: float, duration: float):
    in_edges, out_edges = build_adjacency_index(g)
    nodes_to_remove = []
    for node in g.get_nodes():
        if out_edges.get(node.get_name(), []) != []:
            continue
        if "()" in node.get_label() and "main()" not in node.get_label():            
            total_time = totaltimes.get(node.get_name(), 0)
            if total_time < total_time_limit * duration:
                nodes_to_remove.append(node.get_name())
        
    for node_name in nodes_to_remove:
        remove_node(g, node_name, in_edges, out_edges)

def subtree_pruning(g: Dot, total_time_limit: float, duration: float, children: defaultdict[str, set]):
    prunable_nodes = filter_prunable_nodes(g, total_time_limit, duration, children)

    in_edges, out_edges = build_adjacency_index(g)

    print("number of prunable calls:", len(prunable_nodes))
    print("Total nodes before pruning:", len(g.get_nodes()))

    for call_node in prunable_nodes:
        replace_node(g, call_node.get_name(), call_node.get_name() + "_pruned", in_edges, out_edges)
    
    remove_orphaned_nodes(g)

    prune_leafs(g, total_time_limit, duration)

    remove_orphaned_nodes(g)

    print("Total nodes after pruning:", len(g.get_nodes()))


def color_graph(g: Dot, total_time_limit: float, color: str = "purple", top_n: int = 5):
    selftime_ordered_nodes = []
    for node in g.get_nodes():
        time = totaltimes.get(node.get_name(), 0)
        if time > 0:
            selftime_ordered_nodes.append((time, node))
    selftime_ordered_nodes.sort(reverse=True, key=lambda x: x[0])

    for _, node in selftime_ordered_nodes[:top_n]:
        g.get_node(node.get_name())[0].set_style("filled")
        g.get_node(node.get_name())[0].set_fillcolor(color)

    return selftime_ordered_nodes[:top_n]

def remove_subnodes(g: Dot):
    subnode_edges = []
    for edge in g.get_edges():
        if ":" in edge.get_source() or ":" in edge.get_destination():
            subnode_edges.append(edge)
    for edge in subnode_edges:
        g.del_edge(edge.get_source(), edge.get_destination())
        g.add_edge(pydot.Edge(edge.get_source().split(":")[0], edge.get_destination().split(":")[0], label=edge.get_label(), dir=edge.get_dir()))

def clear_redundant_edges(g: Dot):
    unique_edges = set()
    for edge in g.get_edges():
        edge_tuple = (edge.get_source(), edge.get_destination(), edge.get_label(), edge.get_dir())
        if edge_tuple in unique_edges:
            continue
        else:
            unique_edges.add(edge_tuple)
    for edge in g.get_edges():
        g.del_edge(edge.get_source(), edge.get_destination())
    for edge_tuple in unique_edges:
        if edge_tuple[3] is None:
            g.add_edge(pydot.Edge(edge_tuple[0], edge_tuple[1], label=edge_tuple[2]))
        else:
            g.add_edge(pydot.Edge(edge_tuple[0], edge_tuple[1], label=edge_tuple[2], dir=edge_tuple[3]))

def remove_empty_records(g: Dot):
    for node in g.get_nodes():
        label = node.get_label()
        max_entries = label.count("|")
        if max_entries == 0:
            continue
        for i in range(max_entries):
            label = label.replace(f"|<e{i}> \ ", "")
        node.set_label(label)

def visualize_graph(g: Dot, output_dir: str, filename: str):
    remove_subnodes(g)
    clear_redundant_edges(g)
    remove_empty_records(g)
    os.makedirs(output_dir, exist_ok=True)
    g.write_raw(f"{output_dir}/{filename}")

def calculate_selftimes(input_stream: str) -> Tuple[defaultdict[str, float], float]:
    """
    Parses perf script output, auto-detects frequency, 
    and calculates self-time per function.
    """
    
    # Dictionary to store sample counts: { "function_name": count }
    selftime_counts = defaultdict(int)
    
    # 1. Regex for the Header Line to capture Timestamp
    # Looks for a floating point number followed immediately by a colon
    # Matches: " ... 513845.025989: ..."
    header_pattern = re.compile(r'\s+([0-9]+\.[0-9]+):')

    # 2. Regex for the Stack Frame (Function Name)
    # Matches: "     7ffff... function_name+0xoffset (...)"
    stack_pattern = re.compile(r'^\s*[0-9a-f]+\s+(.+?)(?:\+0x[0-9a-f]+)?\s+\(.+\)')
    
    # State tracking
    looking_for_stack_tip = False
    
    # Frequency detection variables
    first_timestamp = None
    last_timestamp = None
    total_samples_detected = 0
    
    for line in input_stream:
        line = line.rstrip()
        if not line:
            continue

        # --- Check if line is a Stack Frame (indented) ---
        if line[0].isspace():
            if looking_for_stack_tip:
                match = stack_pattern.match(line)
                if match:
                    func_name = match.group(1).strip()
                    selftime_counts[func_name] += 1
                
                # We found the tip; stop looking until next header
                looking_for_stack_tip = False
        
        # --- Check if line is a Header (not indented) ---
        else:
            # It is a header line. Try to find the timestamp.
            ts_match = header_pattern.search(line)
            if ts_match:
                timestamp = float(ts_match.group(1))
                
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp
                
                total_samples_detected += 1
                
                # A header implies a new sample follows; enable flag
                looking_for_stack_tip = True

    # --- Calculate Frequency ---
    if total_samples_detected < 2 or first_timestamp == last_timestamp:
        # Fallback if we don't have enough data to calculate duration
        calculated_frequency = 99.0 
        duration = 0
        print("Warning: Not enough samples to calculate frequency. Defaulting to 99 Hz.")
    else:
        duration = last_timestamp - first_timestamp
        # Frequency = Total Samples / Duration in Seconds
        calculated_frequency = total_samples_detected / duration
    
    sorted_stats = sorted(selftime_counts.items(), key=lambda x: x[1], reverse=True)

    selftimes = defaultdict(float)
    
    for func, count in sorted_stats:
        # Time = Count / Frequency
        time_seconds = count / calculated_frequency
        selftimes[func] = time_seconds

    return selftimes, duration

def calculate_totaltimes(input_stream: str) -> Tuple[defaultdict[str, float], float]:
    """
    Parses perf script output and calculates TOTAL (inclusive) time per function.
    
    Total Time = The duration a function was present anywhere in the call stack.
    """
    
    # Dictionary to store sample counts: { "function_name": count }
    total_time_counts = defaultdict(int)
    
    # Set to track functions in the CURRENT sample/stack
    current_sample_stack = set()
    
    # 1. Regex for the Header Line
    header_pattern = re.compile(r'\s+([0-9]+\.[0-9]+):')

    # 2. Regex for the Stack Frame
    stack_pattern = re.compile(r'^\s*[0-9a-f]+\s+(.+?)(?:\+0x[0-9a-f]+)?\s+\(.+\)')
    
    # Frequency detection variables
    first_timestamp = None
    last_timestamp = None
    total_samples_detected = 0
    
    # Helper to commit the current stack to the global totals
    def commit_current_sample():
        for func_name in current_sample_stack:
            total_time_counts[func_name] += 1
        current_sample_stack.clear()

    for line in input_stream:
        line = line.rstrip()
        if not line:
            continue

        # --- Check if line is a Stack Frame (indented) ---
        if line[0].isspace():
            # We are inside a stack trace; capture every function
            match = stack_pattern.match(line)
            if match:
                func_name = match.group(1).strip()
                # Use a set to handle recursion (count function once per sample)
                current_sample_stack.add(func_name)
        
        # --- Check if line is a Header (not indented) ---
        else:
            # A new header means the previous sample is finished.
            # Commit the functions found in the previous stack (if any).
            if current_sample_stack:
                commit_current_sample()

            # Process the timestamp
            ts_match = header_pattern.search(line)
            if ts_match:
                timestamp = float(ts_match.group(1))
                
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp
                
                total_samples_detected += 1

    # --- End of Stream: Commit the final sample ---
    if current_sample_stack:
        commit_current_sample()

    # --- Calculate Frequency ---
    if total_samples_detected < 2 or first_timestamp == last_timestamp:
        calculated_frequency = 99.0 
        duration = 0
        print("Warning: Not enough samples to calculate frequency. Defaulting to 99 Hz.")
    else:
        duration = last_timestamp - first_timestamp
        calculated_frequency = total_samples_detected / duration
    
    # --- Convert Counts to Seconds ---
    sorted_stats = sorted(total_time_counts.items(), key=lambda x: x[1], reverse=True)
    totaltimes = defaultdict(float)
    
    for func, count in sorted_stats:
        time_seconds = count / calculated_frequency
        totaltimes[func] = time_seconds

    return totaltimes, duration

def calculate_children(input_stream: str) -> Dict[str, set[str]]:
    descendant_map = defaultdict(set)
    stack_pattern = re.compile(r'^\s*[0-9a-f]+\s+(.+?)(?:\+0x[0-9a-f]+)?\s+\(.+\)')

    current_stack = []

    def commit_stack():
        # The stack is captured as [innermost, caller, ..., main]
        # We need to reverse it to [main, ..., caller, innermost] to map parent -> children
        reversed_stack = current_stack[::-1]
        for i in range(len(reversed_stack)):
            parent_func = reversed_stack[i]
            for j in range(i + 1, len(reversed_stack)):
                child_func = reversed_stack[j]
                descendant_map[parent_func].add(child_func)
        current_stack.clear()

    for line in input_stream:
        line = line.rstrip()
        if not line:
            continue

        if line[0].isspace():
            match = stack_pattern.match(line)
            if match:
                func_name = match.group(1).strip()
                current_stack.append(func_name)
        else:
            if current_stack:
                commit_stack()

    # Commit the final stack if the file ends with a stack trace
    if current_stack:
        commit_stack()

    return descendant_map

def parse_profiles(perf_trace: str) -> Tuple[defaultdict[str, float], defaultdict[str, float], float]:
    cmd_mangled = f"perf script --no-demangle -i {perf_trace} -F +srcline --full-source-path > mangled.profile"
    cmd_demangled = f"perf script -i {perf_trace} -F +srcline --full-source-path > demangled.profile"
    os.system(cmd_mangled)
    os.system(cmd_demangled)

    with open("mangled.profile", "r") as f:
        mangled_content = f.readlines()
        mangled_times_total, duration_mangled_total = calculate_totaltimes(mangled_content)
        mangled_times_self, duration_mangled_self = calculate_selftimes(mangled_content)
        mangled_children = calculate_children(mangled_content)
        
    with open("demangled.profile", "r") as f:
        demangled_content = f.readlines()
        demangled_times_total, duration_demangled_total = calculate_totaltimes(demangled_content)
        demangled_times_self, duration_demangled_self = calculate_selftimes(demangled_content)
        demangled_children = calculate_children(demangled_content)

    duration = max(duration_mangled_total, duration_demangled_total, duration_mangled_self, duration_demangled_self)
    
    combined_times_self = defaultdict(float)
    for func in set(mangled_times_self.keys()).union(set(demangled_times_self.keys())):
        combined_times_self[func] = max(mangled_times_self.get(func, 0.0), demangled_times_self.get(func, 0.0))
    
    combined_times_total = defaultdict(float)
    for func in set(mangled_times_total.keys()).union(set(demangled_times_total.keys())):
        combined_times_total[func] = max(mangled_times_total.get(func, 0.0), demangled_times_total.get(func, 0.0))

    children = defaultdict(set)
    for func in set(mangled_children.keys()).union(set(demangled_children.keys())):
        children[func] = mangled_children.get(func, set()).union(demangled_children.get(func, set()))

    os.remove("mangled.profile")
    os.remove("demangled.profile")

    return combined_times_total, combined_times_self, duration, children

def annotate_times(g: Dot, selftimes: defaultdict[str, float], totaltimes: defaultdict[str, float]):
    for node in g.get_nodes():
        label = node.get_label()

        # Only annotate nodes whose label contains '()'
        if "()" not in label:
            continue
        func_name = label.split("()")[0].split()[-1]
            
        if func_name in selftimes:
            time_val_self = selftimes[func_name]
            self_str = f"| Self-time: {time_val_self:.6f}s|"
        else:
            self_str = ""
        if func_name in totaltimes:
            time_val_total = totaltimes[func_name]
            total_str = f"| Total-time: {time_val_total:.6f}s|"
        else:
            total_str = ""

        new_label = re.sub(rf"\b{re.escape(func_name)}\b", f"{self_str} {total_str} {func_name}", label, count=1)
        node.set_label(new_label)
        node.set('shape', 'record')

def total_time(g: Dot) -> float:
    total = 0.0
    root = g.get_node_list()[0]
    label = root.get_label()
    if "Total-time:" in label:
        time_str = label.split("Total-time:")[1].split("s")[0]
        try:
            time_val = float(time_str)
            if time_val > total:
                total = time_val
        except ValueError:
            pass
    return total

def preprocess_graph(g: Dot, selftimes: defaultdict[str, float], totaltimes: defaultdict[str, float]):
    print("Preprocessing graph...")
    annotate_times(g, selftimes, totaltimes)

def handle_single_cfg(g: Dot, total_time_limit: float, top_n:int, duration: float, children: defaultdict[str, set]):
    subtree_pruning(g, total_time_limit, duration, children)
    
    color_graph(g, total_time_limit=total_time_limit, color="pink", top_n=top_n)

def read_graphs(matches: list[str]) -> list[Tuple[str, Dot]]:
    base_graphs = []
    for path in tqdm(matches, desc="Reading DOT files"):
        g = read_dot_file(path)
        base_graphs.append((path, g))

    return base_graphs        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune and analyze a Graphviz DOT file representing a CFG.")
    parser.add_argument("path", type=str, help="Path to .dot file or directory containing .main.cfg.dot files")
    parser.add_argument("perf_trace", type=str, help="Path to perf trace file")
    parser.add_argument("--total_time_limit", type=float, default=0.01, help="Fraction of total self-time to use as pruning threshold (default: 0.01)")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top nodes by self-time to highlight (default: 5)")
    parser.add_argument("--output", type=str, default="cfgs_annotated", help="Output directory for annotated CFGs (default: cfgs_annotated)")
    parser.add_argument("--entry-points", type=str, help="Dumps a list of relevant entry points to the specified file")
    
    args = parser.parse_args()
    
    path = args.path
    total_time_limit = args.total_time_limit
    top_n = args.top_n
    perf_trace = args.perf_trace
    output_dir = args.output

    global totaltimes

    totaltimes, selftimes, duration, children = parse_profiles(perf_trace)

    print("Total runtime: " + str(duration))

    sorted_totaltimes = sorted(totaltimes.items(), key=lambda x: x[1], reverse=True)
    print("Top total times:")
    for func, time in sorted_totaltimes:
        if time <= total_time_limit * duration:
            break
        print(f"{func}: {time:.6f}s")
        if args.entry_points:
            mode = "w" if func == sorted_totaltimes[0][0] else "a"
            with open(args.entry_points, mode) as f:
                f.write(f"{func}\n")

    if not os.path.exists(path):
        print("File not found:", path)
        sys.exit(2)
    
    if os.path.isdir(path):
        matches = []
        for root, _, filenames in os.walk(path):
            for fname in filenames:
                if fname.endswith(".cfg.dot"):
                    matches.append(os.path.join(root, fname))

        if not matches:
            print("No .cfg.dot files found in", path)
            sys.exit(3)

        # use the first match as the dot file to process
        if len(matches) > 1:
            print("Multiple .cfg.dot files found: ", len(matches))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    unique_graphs = read_graphs(matches)
    
    for path, g in unique_graphs:
        filename = path.split("/")[-1]
        preprocess_graph(g, selftimes, totaltimes)
        if total_time(g) <= total_time_limit * duration:
            print("Skipping ", path, " due to low total total-time.")
            continue

        handle_single_cfg(g, total_time_limit, top_n, duration, children)

        visualize_graph(g, output_dir, filename)
    