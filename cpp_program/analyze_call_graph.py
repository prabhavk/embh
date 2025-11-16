#!/usr/bin/env python3
"""
Analyze the call graph and dependency graph of the Manager constructor
in embh_core.cpp. This script parses the C++ code to identify:
1. All methods called from the Manager constructor
2. All methods those methods call (transitive dependencies)
3. All class members accessed
4. All types/classes used
"""

import re
from collections import defaultdict, deque
from pathlib import Path

def read_file(filepath):
    """Read file contents."""
    with open(filepath, 'r') as f:
        return f.read()

def find_method_definition(code, class_name, method_name):
    """Find the start and end line numbers of a method definition."""
    lines = code.split('\n')

    # Pattern to match method definition start
    # Handle various formats: void Class::method(...) { or Class::Class(...) {
    if method_name == class_name:
        # Constructor pattern
        pattern = rf'^{class_name}::{method_name}\s*\('
    else:
        # Regular method pattern
        pattern = rf'^\s*\w+[\s\*&]*\s+{class_name}::{method_name}\s*\('

    start_line = None
    brace_count = 0
    in_method = False

    for i, line in enumerate(lines):
        if start_line is None and re.search(pattern, line):
            start_line = i + 1  # 1-indexed
            # Check if opening brace is on same line
            if '{' in line:
                brace_count = line.count('{') - line.count('}')
                in_method = True
        elif start_line is not None and not in_method:
            # Look for opening brace
            if '{' in line:
                brace_count = line.count('{') - line.count('}')
                in_method = True
        elif in_method:
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                return start_line, i + 1  # end line (1-indexed)

    return start_line, None

def extract_method_body(code, class_name, method_name):
    """Extract the complete method body."""
    start, end = find_method_definition(code, class_name, method_name)
    if start and end:
        lines = code.split('\n')
        return '\n'.join(lines[start-1:end])
    return None

def find_method_calls_in_body(body, class_name="SEM"):
    """Find all method calls in a method body."""
    calls = set()

    # Pattern for this->MethodName() or MethodName() after this->
    patterns = [
        rf'this->\s*({class_name}::)?(\w+)\s*\(',  # this->method() or this->Class::method()
        rf'{class_name}::\s*(\w+)\s*\(',  # Class::method()
        rf'(\w+)\s*->\s*(\w+)\s*\(',  # obj->method()
    ]

    # Find direct method calls
    for match in re.finditer(r'this\s*->\s*(\w+)\s*\(', body):
        method = match.group(1)
        # Filter out common non-method names
        if not method.startswith(('push_back', 'clear', 'insert', 'find', 'begin', 'end',
                                   'size', 'at', 'empty', 'reserve', 'resize', 'swap')):
            calls.add(method)

    # Find static method calls
    for match in re.finditer(rf'{class_name}::(\w+)\s*\(', body):
        calls.add(match.group(1))

    return calls

def find_member_accesses(body):
    """Find all member variable accesses in a method body."""
    members = set()

    # Pattern for this->member
    for match in re.finditer(r'this\s*->\s*(\w+)(?!\s*\()', body):
        member = match.group(1)
        # Filter out common STL method names
        if not match.group(0).strip().endswith('('):
            members.add(member)

    return members

def find_all_method_definitions(code):
    """Find all method definitions in the code."""
    methods = {}

    # Pattern: ReturnType ClassName::MethodName(...)
    pattern = r'^(?:\w+[\s\*&]*\s+)?(\w+)::(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?\{'

    lines = code.split('\n')
    for i, line in enumerate(lines):
        match = re.match(pattern, line)
        if match:
            class_name = match.group(1)
            method_name = match.group(2)
            if class_name not in methods:
                methods[class_name] = {}
            methods[class_name][method_name] = i + 1  # 1-indexed

    return methods

def find_class_definitions(code):
    """Find all class definitions and their members."""
    classes = {}

    lines = code.split('\n')
    in_class = None
    brace_count = 0

    for i, line in enumerate(lines):
        # Start of class
        if in_class is None:
            match = re.match(r'^class\s+(\w+)\s*(?::\s*public\s+\w+\s*)?\{', line)
            if match:
                in_class = match.group(1)
                classes[in_class] = {'start': i+1, 'members': [], 'methods': []}
                brace_count = 1
            elif re.match(r'^class\s+(\w+)', line):
                match = re.match(r'^class\s+(\w+)', line)
                in_class = match.group(1)
                classes[in_class] = {'start': i+1, 'members': [], 'methods': []}
                brace_count = 0
        else:
            brace_count += line.count('{') - line.count('}')

            # Parse members and methods
            # Member variables: type name;
            member_match = re.match(r'\s+(?:[\w:<>,\s\*&]+)\s+(\w+)\s*(?:=|;)', line)
            if member_match and '(' not in line:
                classes[in_class]['members'].append(member_match.group(1))

            # Method declarations: type name(...);
            method_match = re.match(r'\s+(?:virtual\s+)?(?:[\w\*&:<>\s]+)\s+(\w+)\s*\(', line)
            if method_match:
                classes[in_class]['methods'].append(method_match.group(1))

            if brace_count == 0 and in_class:
                classes[in_class]['end'] = i+1
                in_class = None

    return classes

def build_call_graph(code, entry_class, entry_method):
    """Build complete call graph starting from entry point."""
    all_methods = find_all_method_definitions(code)

    call_graph = defaultdict(set)
    member_accesses = defaultdict(set)
    visited = set()
    queue = deque([(entry_class, entry_method)])

    while queue:
        class_name, method_name = queue.popleft()

        if (class_name, method_name) in visited:
            continue
        visited.add((class_name, method_name))

        body = extract_method_body(code, class_name, method_name)
        if body:
            # Find method calls
            calls = find_method_calls_in_body(body, class_name)
            call_graph[(class_name, method_name)] = calls

            # Find member accesses
            members = find_member_accesses(body)
            member_accesses[(class_name, method_name)] = members

            # Queue up called methods for analysis
            for called_method in calls:
                if class_name in all_methods and called_method in all_methods[class_name]:
                    queue.append((class_name, called_method))
                # Also check if it's calling methods on member objects
                # This is a simplification - in reality would need type analysis

    return call_graph, member_accesses, visited

def analyze_manager_constructor():
    """Main analysis function."""
    code_path = Path(__file__).parent / "embh_core.cpp"
    code = read_file(code_path)

    print("=" * 70)
    print("CALL GRAPH ANALYSIS FOR MANAGER CONSTRUCTOR")
    print("=" * 70)

    # Find the manager constructor
    print("\n1. Finding Manager constructor...")
    manager_body = extract_method_body(code, "manager", "manager")
    if not manager_body:
        print("ERROR: Could not find manager constructor")
        return

    print(f"   Found manager constructor ({len(manager_body.split(chr(10)))} lines)")

    # Build call graph
    print("\n2. Building call graph...")
    call_graph, member_accesses, visited = build_call_graph(code, "manager", "manager")

    print(f"   Found {len(visited)} methods in call graph")

    # Organize by class
    methods_by_class = defaultdict(list)
    for class_name, method_name in sorted(visited):
        methods_by_class[class_name].append(method_name)

    print("\n3. Methods required (organized by class):")
    print("-" * 70)

    all_methods_list = []
    for class_name in sorted(methods_by_class.keys()):
        print(f"\n   {class_name}:")
        for method_name in sorted(methods_by_class[class_name]):
            start, end = find_method_definition(code, class_name, method_name)
            if start and end:
                line_count = end - start + 1
                print(f"      - {method_name} (lines {start}-{end}, {line_count} lines)")
                all_methods_list.append((class_name, method_name, start, end))
            else:
                print(f"      - {method_name} (location unknown)")
                all_methods_list.append((class_name, method_name, None, None))

    # Find all member variables accessed
    print("\n4. Member variables accessed:")
    print("-" * 70)

    all_members = set()
    for (class_name, method_name), members in member_accesses.items():
        all_members.update(members)

    for member in sorted(all_members):
        print(f"   - {member}")

    # Find direct calls from manager constructor
    print("\n5. Direct method calls from manager constructor:")
    print("-" * 70)

    manager_calls = find_method_calls_in_body(manager_body, "manager")
    for call in sorted(manager_calls):
        print(f"   - {call}")

    # Also check what P (SEM object) methods are called
    print("\n6. SEM methods called via this->P->:")
    print("-" * 70)

    sem_calls = set()
    for match in re.finditer(r'this\s*->\s*P\s*->\s*(\w+)\s*\(', manager_body):
        sem_calls.add(match.group(1))

    for call in sorted(sem_calls):
        print(f"   - {call}")

    # Recursively find all SEM methods
    print("\n7. Complete SEM method dependency tree:")
    print("-" * 70)

    sem_call_graph, sem_members, sem_visited = build_call_graph(code, "SEM", "")

    # Start from the methods called in manager constructor
    queue = deque()
    for method in sem_calls:
        queue.append(("SEM", method))

    sem_all_methods = set()
    while queue:
        class_name, method_name = queue.popleft()
        if (class_name, method_name) in sem_all_methods:
            continue
        sem_all_methods.add((class_name, method_name))

        body = extract_method_body(code, class_name, method_name)
        if body:
            calls = find_method_calls_in_body(body, class_name)
            for call in calls:
                queue.append((class_name, call))

    sem_methods_sorted = []
    for class_name, method_name in sorted(sem_all_methods):
        start, end = find_method_definition(code, class_name, method_name)
        if start and end:
            sem_methods_sorted.append((method_name, start, end, end-start+1))
            print(f"   - {method_name} (lines {start}-{end})")
        else:
            print(f"   - {method_name} (not found - may be inline or in header)")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_lines = 0
    for _, _, start, end in sem_methods_sorted:
        if start and end:
            total_lines += end - start + 1

    print(f"Total SEM methods required: {len(sem_all_methods)}")
    print(f"Total lines of code (approximate): {total_lines}")
    print(f"Member variables accessed: {len(all_members)}")

    # Write results to file
    output_file = Path(__file__).parent / "call_graph_analysis.txt"
    with open(output_file, 'w') as f:
        f.write("MANAGER CONSTRUCTOR CALL GRAPH ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("SEM METHODS REQUIRED (sorted by line number):\n")
        f.write("-" * 70 + "\n")

        for method, start, end, lines in sorted(sem_methods_sorted, key=lambda x: x[1] if x[1] else 0):
            f.write(f"{method}: lines {start}-{end} ({lines} lines)\n")

        f.write(f"\nTOTAL: {len(sem_methods_sorted)} methods, ~{total_lines} lines\n")

        f.write("\n\nMEMBER VARIABLES:\n")
        f.write("-" * 70 + "\n")
        for member in sorted(all_members):
            f.write(f"  {member}\n")

    print(f"\nResults written to: {output_file}")

    return sem_methods_sorted, all_members

if __name__ == "__main__":
    analyze_manager_constructor()
