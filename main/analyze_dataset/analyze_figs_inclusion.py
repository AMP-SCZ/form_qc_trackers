import pandas as pd
import os
import sys
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
import json
import re
from collections import defaultdict
from graphviz import Digraph

# if([chrpsychs_scr_ac7]=1 or [chrpsychs_scr_ac27]=1,1,if(([chrpsychs_scr_ac7]<>''
# and [chrpsychs_scr_ac7]=0) and ([chrpsychs_scr_ac27]<>'' and [chrpsychs_scr_ac27]=0),0,''))

# psychs vars that pull from figs
# chrpsychs_scr_e2
# chrpsychs_fu_e2
# hcpsychs_fu_e2

# exclusion criteria chrcrit_excl5 pulls from chrpsychs_scr_ac1


# plan 

# 1. find figs variables that connect to psychs form variables and create a recursive tree of all variables affected by those
# of all variables that they connect to up until the inclusion status form

# 2. find which participants were included only because of the value in those variables 

# 3. determine which participants would change cohorts based on figs edits

class AnalyzeFigs():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.depen_path = self.config_info['paths']['dependencies_path']
        self.output_path = self.config_info['paths']['output_path']
        self.data_dict_df = self.utils.read_data_dictionary()

    def run_script(self):
        #self.recursive_filter_df(self.data_dict_df)
        self.filter_inclusion_status()

    def filter_inclusion_status(self):
        for network in ['PRONET','PRESCIENT']:
            screening_df =  pd.read_csv((f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
            f'screening_{network}-day1to1.csv'), keep_default_na = False)
            for row in screening_df.itertuples():    
                self.isolate_figs_inclusion_reasons(row)

    def isolate_figs_inclusion_reasons(self, row):
        if all(var_value in self.utils.all_dtype([1])
        for var_value in [row.chrpsychs_scr_ac6, 
        row.chrpsychs_scr_ac7, row.chrpsychs_scr_e2]):
            if (row.chrpsychs_scr_e2 in self.utils.all_dtype([1])
            and row.chrpsychs_scr_e1 not in self.utils.all_dtype([1])):
                if not any(var_val in self.utils.all_dtype([1]) for
                var_val in [row.chrpsychs_scr_ac2,
                row.chrpsychs_scr_ac3,row.chrpsychs_scr_ac4]):
                    print(row.subjectid)

    def search_figs_vars(self):
        pass

    def filter_df_by_list(self, filter_list, inp_df, filtered_col):
        output_df = inp_df[inp_df[
        filtered_col].str.contains("|".join(map(re.escape, filter_list)))]

        return output_df

    def recursive_filter_df(self, inp_df):
        # find any variables that chrpsychs_scr_e2
        # feeds into that is connected to
        # chrcrit_inc3 in any way
        filtered_data_dict = self.data_dict_df[self.data_dict_df[
        'Choices, Calculations, OR Slider Labels'].str.contains('chrpsychs_scr_e2')]

        print(filtered_data_dict)

        psychs_vars = filtered_data_dict['Variable / Field Name'].tolist()
        print('-------')
        print(psychs_vars)

        filtered_df = self.filter_df_by_list(psychs_vars, self.data_dict_df,
        'Choices, Calculations, OR Slider Labels')

        branched_vars = filtered_df['Variable / Field Name'].tolist()
        count = 0

        while not any('chrcrit_inc3' in var for var in branched_vars):
        # while count < 100:
            count += 1
            print(count)
            print(branched_vars)
            filtered_df = self.filter_df_by_list(branched_vars, self.data_dict_df,
            'Choices, Calculations, OR Slider Labels')
            branched_vars = filtered_df['Variable / Field Name'].tolist()
            print(branched_vars)

    def determine_figs_included_participants(self):
        pass

class RedcapDependencyAnalyzer:
    """
    A class to analyze variable dependencies in a REDCap data dictionary.
    
    It builds a directed dependency graph based on variable references in 
    calculation fields and lets you trace relationships between forms or variables.
    """

    def __init__(self, csv_path, calc_col="select_choices_or_calculations",
    field_col="Variable / Field Name", type_col="Field Type"):
        """
        Initialize the analyzer.

        Args:
            csv_path (str): Path to the REDCap data dictionary CSV.
            calc_col (str): Column containing calculation expressions.
            field_col (str): Column containing variable (field) names.
            type_col (str): Column specifying the field type (e.g., 'calc').
        """
        self.csv_path = csv_path
        self.calc_col = calc_col
        self.field_col = field_col
        self.type_col = type_col

        self.df = pd.read_csv(csv_path, dtype=str).fillna("")
        
        self.forward_graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)

        self.pattern = re.compile(r'\[([^\]]+)\]')
        
        self._build_graphs()

    def _build_graphs(self):
        """Parse calculation
        expressions and build dependency graphs."""
        for _, row in self.df.iterrows():
            field = row[self.field_col]
            calc_expr = row.get(self.calc_col, "")
            field_type = row.get(self.type_col, "")

            # Only consider calculation fields
            if field_type == "calc" and calc_expr:
                refs = self.pattern.findall(calc_expr)
                for ref in refs:
                    self.forward_graph[ref].add(field)
                    self.reverse_graph[field].add(ref)

    def reachable_from_sources(self, sources, graph):
        """Return all nodes reachable
        from a set of sources in the given graph."""
        visited = set()
        stack = list(sources)
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nbr in graph.get(node, []):
                if nbr not in visited:
                    stack.append(nbr)
        return visited

    def extract_branch(self, source_vars, target_vars):
        """
        Find all variables that lie on any path from sources to targets.

        Args:
            source_vars (list[str]): Starting variables (e.g., FIGS fields).
            target_vars (list[str]): Target variables (e.g., inclusion fields).

        Returns:
            set[str]: All variable names on paths connecting sources to targets.
        """
        downstream = self.reachable_from_sources(source_vars, self.forward_graph)
        upstream = self.reachable_from_sources(target_vars, self.reverse_graph)

        branch_nodes = downstream & upstream
        branch_nodes |= set(source_vars) | set(target_vars)
        return branch_nodes

    def print_tree(self, root, allowed_nodes, targets, indent=0, visited=None):
        """Recursively print a dependency tree for a root variable."""
        if visited is None:
            visited = set()
        prefix = " " * indent
        label = root
        if root in targets:
            label += "  <-- inclusion target"
        print(prefix + label)

        if root in visited:
            print(prefix + "  (cycle detected)")
            return
        visited.add(root)

        for child in sorted(self.forward_graph.get(root, [])):
            if child in allowed_nodes:
                self.print_tree(child, allowed_nodes, targets, indent + 2, visited.copy())

    def print_branches(self, source_vars, target_vars):
        """Print all dependency trees
        from each source variable toward target variables."""
        branch_nodes = self.extract_branch(source_vars, target_vars)
        for src in source_vars:
            if src in branch_nodes:
                print(f"\nTree from: {src}")
                self.print_tree(src, branch_nodes, set(target_vars))

    def find_paths(self, sources, targets):
        """Return all possible paths from
        any source to any target within the branch."""
        allowed_nodes = self.extract_branch(sources, targets)
        all_paths = []
        targets = set(targets)

        def dfs(node, path):
            path = path + [node]
            if node in targets:
                all_paths.append(path)
                return
            for nbr in self.forward_graph.get(node, []):
                if nbr in allowed_nodes and nbr not in path:
                    dfs(nbr, path)

        for src in sources:
            if src in allowed_nodes:
                dfs(src, [])
        return all_paths

if __name__ == '__main__':
    def save_dependency_graph_png(edges, roots, target=None, filename="psychs_dependencies"):
        """
        edges: dict[parent] = set(children)
        roots: list of root variables (your psychs_vars)
        target: optional variable name to highlight (e.g., 'chrcrit_inc3')
        filename: output file base name (without extension)
        """

        dot = Digraph(format='png')
        dot.attr(rankdir='LR')  # left-to-right; use 'TB' for top-to-bottom

        # Add nodes
        all_nodes = set(edges.keys())
        for children in edges.values():
            all_nodes.update(children)

        for node in all_nodes:
            if node in roots:
                # style psychs roots
                dot.node(node, shape='box', style='filled', fillcolor='lightgray')
            elif target is not None and node == target:
                # style target (e.g., inclusion var)
                dot.node(node, shape='doublecircle', style='filled', fillcolor='lightgreen')
            else:
                dot.node(node, shape='ellipse')

        # Add edges
        for parent, children in edges.items():
            for child in children:
                dot.edge(parent, child)

        # Render to PNG (creates filename.png)
        output_path = dot.render(filename, cleanup=True)
        print(f"Saved graph to: {output_path}")


    psychs_vars = ['chrpsychs_scr_e2']
    data_dict_path = "/home/ob001/dependencies/data_dictionary/current_data_dictionary.csv"
    data_dict_df = pd.read_csv(data_dict_path, keep_default_na = False)
    filtered_df = data_dict_df[data_dict_df[
    'Variable / Field Name'].isin([
    'chrpsychs_scr_e2','chrpsychs_fu_e2','hcpsychs_fu_e2'])]
    equations = filtered_df['Choices, Calculations, OR Slider Labels'].tolist()
    figs_vars = []
    for equation in equations:
        splt_list = equation.split('[')
        for eq_part in splt_list:
            for var in eq_part.split(']'):
                if 'chrfigs' in var and var not in figs_vars:
                    figs_vars.append(var)
    def check_var_presence(input_equation, match_var):
        splt_list = input_equation.split('[')
        for eq_part in splt_list:
            for var in eq_part.split(']'):
                if var == match_var:
                    return True
        return False

    print(figs_vars)
    data_dict_df = data_dict_df.rename(columns={
        "Variable / Field Name": "variable",
        "Choices, Calculations, OR Slider Labels": "calculation"
    })

    connected_vars = set(psychs_vars)      
    to_visit = list(psychs_vars) 
    to_visit = ['chrfigs_father_napdef2']          
    target = "chrfigs_father_napdx"                

    edges = defaultdict(set)

    while to_visit:
        current = to_visit.pop(0) 

        for row in data_dict_df.itertuples():
            calc = row.calculation
            if not calc:
                continue

            if check_var_presence(calc, current) and row.variable not in connected_vars:
                connected_vars.add(row.variable)
                to_visit.append(row.variable)
                edges[current].add(row.variable)

                print(f"Discovered: {row.variable}, total connected: {len(connected_vars)}")

    # 3. Tree printing function for visualization
    def print_tree(root, graph, target=None, indent=0, visited=None):
        if visited is None:
            visited = set()

        prefix = " " * indent
        label = root
        if target is not None and root == target:
            label += "  <-- TARGET"
        print(prefix + label)

        # prevent infinite loops on cycles
        if root in visited:
            print(prefix + "  (cycle detected)")
            return
        visited.add(root)

        for child in sorted(graph.get(root, [])):
            print_tree(child, graph, target=target, indent=indent + 2, visited=visited.copy())

    # 4. Visualize: print a tree from each psychs seed
    print("\n===== DEPENDENCY TREES FROM PSYCHS VARS =====")
    for src in psychs_vars:
        print(f"\nTree from: {src}")
        if src in connected_vars:
            print_tree(src, edges, target=target)
        else:
            print(f"{src} (no dependents found)")

    save_dependency_graph_png(edges, roots=psychs_vars, target='chrcrit_included',
                          filename="psychs_dependencies")


    def generate_reverse_tree_png(data_dict_df, target="chrcrit_inc3", filename="chrcrit_inc3_ancestors"):
        """
        Build a reverse dependency tree ending at `target` and save it as a PNG.

        Requirements:
        - data_dict_df must have columns: 'variable' and 'calculation'
        - Graphviz must be installed (system + `pip install graphviz`)
        """

        # --- 1. Build reverse_edges: child -> set(parents) ---
        var_pattern = re.compile(r'\[([^\]]+)\]')
        reverse_edges = defaultdict(set)

        for row in data_dict_df.itertuples():
            var = row.variable
            calc = row.calculation
            if not calc:
                continue

            refs = var_pattern.findall(calc)
            for ref in refs:
                if ref == var:
                    continue  # ignore self-ref
                # ref ----> var
                reverse_edges[var].add(ref)

        # --- 2. Collect all ancestors of target via BFS ---
        ancestors = set()
        to_visit = [target]

        while to_visit:
            current = to_visit.pop(0)
            for parent in reverse_edges.get(current, []):
                if parent not in ancestors:
                    ancestors.add(parent)
                    to_visit.append(parent)

        if not ancestors:
            print(f"No ancestors found for target '{target}'.")
            return None

        # --- 3. Build forward subgraph (parent -> child) restricted to target + ancestors ---
        forward_subgraph = defaultdict(set)
        for child, parents in reverse_edges.items():
            if child == target or child in ancestors:
                for p in parents:
                    if p in ancestors or child == target:
                        forward_subgraph[p].add(child)

        dot = Digraph(format='png')
        dot.attr(rankdir='LR')  # top-to-bottom layout


        all_nodes = set(ancestors) | {target}
        for node in all_nodes:
            if node == target:
                dot.node(node, shape='doublecircle', style='filled', fillcolor='lightgreen')
            else:
                dot.node(node, shape='ellipse')

        for parent, children in forward_subgraph.items():
            for child in children:
                dot.edge(parent, child)

        output_path = dot.render(filename, cleanup=True)
        print(f"Saved ancestor tree PNG to: {output_path}")
        return output_path


    generate_reverse_tree_png(data_dict_df, target="chrcrit_included", filename="chrcrit_included_ancestors")

    # 1. Initialize
    """analyzer = RedcapDependencyAnalyzer(data_dict_path)

    # 2. Define your variables
    inclusion_vars = ["data_dict_path"]

    # 3. Print dependency trees
    analyzer.print_branches(figs_vars, inclusion_vars)

    # 4. Or get explicit paths
    paths = analyzer.find_paths(figs_vars, inclusion_vars)
    for p in paths:
        print(" -> ".join(p))"""
