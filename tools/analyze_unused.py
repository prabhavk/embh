#!/usr/bin/env python3
"""
C++ Unused Code Analyzer

Analyzes C++ files to find unused functions and unused variables.
Usage:
  python analyze_unused.py <cpp_file> --functions    # Find unused functions
  python analyze_unused.py <cpp_file> --variables    # Find unused variables
  python analyze_unused.py <cpp_file> --all          # Find both
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Set, List, Dict, Tuple
from collections import defaultdict


class CppUnusedAnalyzer:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.content = self._read_file()
        self.content_no_comments = self._remove_comments(self.content)
        self.content_no_strings = self._remove_string_literals(self.content_no_comments)
    
    def _read_file(self) -> str:
        """Read the C++ file content."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: File '{self.filepath}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    
    def _remove_comments(self, text: str) -> str:
        """Remove C++ comments from text."""
        # Remove single-line comments
        text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
        # Remove multi-line comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text
    
    def _remove_string_literals(self, text: str) -> str:
        """Remove string literals to avoid false positives."""
        text = re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)
        text = re.sub(r"'(?:[^'\\]|\\.)*'", "''", text)
        return text
    
    def _extract_functions(self) -> Dict[str, Dict]:
        """Extract all function definitions."""
        functions = {}
        
        # Pattern for function definitions
        # Matches: return_type function_name(parameters) { ... }
        func_pattern = r'^[\s]*(?:inline\s+|static\s+|virtual\s+|explicit\s+)?(?:[\w:<>,\s*&]+)\s+(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?\s*\{'
        
        lines = self.content_no_comments.split('\n')
        
        for i, line in enumerate(lines, 1):
            match = re.match(func_pattern, line)
            if match:
                func_name = match.group(1)
                params = match.group(2).strip()
                
                # Skip C++ keywords
                cpp_keywords = {
                    'if', 'while', 'for', 'switch', 'catch', 'class', 
                    'struct', 'namespace', 'operator'
                }
                
                if func_name not in cpp_keywords:
                    functions[func_name] = {
                        'line': i,
                        'params': params,
                        'full_line': line.strip()
                    }
        
        return functions
    
    def _extract_class_member_variables(self) -> Dict[str, List[Tuple[str, int]]]:
        """Extract class member variables grouped by class."""
        class_members = defaultdict(list)
        
        lines = self.content_no_comments.split('\n')
        current_class = None
        brace_depth = 0
        in_class = False
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Detect class start
            class_match = re.match(r'^\s*class\s+(\w+)', stripped)
            if class_match:
                current_class = class_match.group(1)
                in_class = True
                brace_depth = 0
            
            if in_class:
                brace_depth += stripped.count('{') - stripped.count('}')
                
                if brace_depth == 0 and '}' in stripped:
                    in_class = False
                    current_class = None
                    continue
                
                # Look for member variable declarations
                # Pattern: type variable_name; (not in a function)
                if current_class and brace_depth == 1:
                    # Skip function definitions
                    if '(' in stripped and ')' in stripped and '{' in stripped:
                        continue
                    
                    # Match variable declarations
                    var_pattern = r'^\s*(?:static\s+|const\s+|mutable\s+)?(?:[\w:<>,\s*&]+)\s+(\w+)\s*[;=]'
                    var_match = re.match(var_pattern, stripped)
                    
                    if var_match:
                        var_name = var_match.group(1)
                        # Skip common C++ keywords that might match
                        if var_name not in {'public', 'private', 'protected', 'class', 'struct'}:
                            class_members[current_class].append((var_name, i))
        
        return class_members
    
    def _extract_local_variables(self) -> List[Tuple[str, int, str]]:
        """Extract local variables from functions."""
        local_vars = []
        
        lines = self.content_no_comments.split('\n')
        in_function = False
        brace_depth = 0
        current_function = None
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Detect function start
            func_match = re.match(r'^[\s]*(?:[\w:<>,\s*&]+)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?\s*\{', line)
            if func_match:
                current_function = func_match.group(1)
                in_function = True
                brace_depth = 1
                continue
            
            if in_function:
                brace_depth += stripped.count('{') - stripped.count('}')
                
                if brace_depth == 0:
                    in_function = False
                    current_function = None
                    continue
                
                # Look for variable declarations inside functions
                # Pattern: type variable_name = ...;  or  type variable_name;
                var_patterns = [
                    r'^\s*(?:const\s+)?(\w+)\s+(\w+)\s*[;=]',  # Simple: int x;
                    r'^\s*(?:const\s+)?(\w+)\s*<[^>]+>\s+(\w+)\s*[;=(]',  # Template: vector<int> x;
                    r'^\s*(?:const\s+)?(\w+)\s*\*\s*(\w+)\s*[;=]',  # Pointer: int* x;
                    r'^\s*(?:const\s+)?(\w+)\s*&\s*(\w+)\s*[;=]',  # Reference: int& x;
                ]
                
                for pattern in var_patterns:
                    var_match = re.match(pattern, stripped)
                    if var_match:
                        var_type = var_match.group(1)
                        var_name = var_match.group(2)
                        
                        # Skip if it's a function call or keyword
                        cpp_keywords = {
                            'if', 'for', 'while', 'switch', 'return', 'delete',
                            'auto', 'void', 'class', 'struct', 'namespace'
                        }
                        
                        if var_name not in cpp_keywords and current_function:
                            local_vars.append((var_name, i, current_function))
                            break
        
        return local_vars
    
    def _is_variable_used(self, var_name: str, scope_start: int = 0, scope_end: int = None) -> bool:
        """Check if a variable is used in the code."""
        if scope_end is None:
            scope_end = len(self.content_no_strings)
        
        lines = self.content_no_strings.split('\n')
        scope_lines = lines[scope_start:scope_end] if scope_end else lines[scope_start:]
        scope_text = '\n'.join(scope_lines)
        
        # Pattern to match variable usage (not declaration)
        # Look for: variable_name followed by . -> [ or in expressions
        usage_patterns = [
            rf'\b{re.escape(var_name)}\s*\.',      # member access: var.member
            rf'\b{re.escape(var_name)}\s*->',      # pointer access: var->member
            rf'\b{re.escape(var_name)}\s*\[',      # array access: var[index]
            rf'\b{re.escape(var_name)}\s*\(',      # function call: var()
            rf'[=+\-*/&|^%]\s*{re.escape(var_name)}\b',  # in expression: = var
            rf'\b{re.escape(var_name)}\s*[+\-*/&|^%=]',  # in expression: var +
            rf'\({re.escape(var_name)}\b',         # function arg: (var)
            rf',\s*{re.escape(var_name)}\b',       # function arg: , var
            rf'return\s+{re.escape(var_name)}\b',  # return statement
            rf'<<\s*{re.escape(var_name)}\b',      # cout << var
            rf'>>\s*{re.escape(var_name)}\b',      # cin >> var
        ]
        
        for pattern in usage_patterns:
            if re.search(pattern, scope_text):
                return True
        
        return False
    
    def find_unused_functions(self) -> List[Dict]:
        """Find all functions that are not called."""
        functions = self._extract_functions()
        
        if not functions:
            return []
        
        # Find which functions are called
        called_functions = set()
        
        for func_name in functions:
            # Pattern to match function calls: function_name(
            call_pattern = rf'\b{re.escape(func_name)}\s*\('
            
            # Count occurrences (subtract 1 for the definition itself)
            occurrences = len(re.findall(call_pattern, self.content_no_strings))
            
            if occurrences > 1:  # More than just the definition
                called_functions.add(func_name)
        
        # Find unused functions
        unused = []
        
        for func_name, details in functions.items():
            # Skip special functions
            if func_name in {'main', 'Main'}:
                continue
            
            # Skip constructors/destructors (would need class name matching)
            if func_name.startswith('~'):
                continue
            
            if func_name not in called_functions:
                unused.append({
                    'name': func_name,
                    'line': details['line'],
                    'params': details['params'],
                    'full_line': details['full_line']
                })
        
        return unused
    
    def find_unused_variables(self) -> Dict[str, List]:
        """Find unused member and local variables."""
        unused = {
            'member': [],
            'local': []
        }
        
        # Check member variables
        class_members = self._extract_class_member_variables()
        
        for class_name, members in class_members.items():
            for var_name, line_num in members:
                # Search for usage in the entire file
                if not self._is_variable_used(var_name, line_num):
                    unused['member'].append({
                        'name': var_name,
                        'class': class_name,
                        'line': line_num
                    })
        
        # Check local variables
        local_vars = self._extract_local_variables()
        
        for var_name, line_num, func_name in local_vars:
            # Search for usage after declaration line
            if not self._is_variable_used(var_name, line_num):
                unused['local'].append({
                    'name': var_name,
                    'function': func_name,
                    'line': line_num
                })
        
        return unused
    
    def print_unused_functions(self, unused_functions: List[Dict]):
        """Print report of unused functions."""
        print("\n" + "=" * 70)
        print("UNUSED FUNCTIONS REPORT")
        print("=" * 70)
        
        if not unused_functions:
            print("\n✓ No unused functions found!")
            print("  All functions are being called.")
        else:
            print(f"\n⚠ Found {len(unused_functions)} unused function(s):\n")
            
            for func in sorted(unused_functions, key=lambda x: x['line']):
                print(f"  • {func['name']}()")
                print(f"    Line: {func['line']}")
                if func['params']:
                    print(f"    Parameters: {func['params']}")
                print(f"    Code: {func['full_line'][:60]}...")
                print()
    
    def print_unused_variables(self, unused_vars: Dict[str, List]):
        """Print report of unused variables."""
        print("\n" + "=" * 70)
        print("UNUSED VARIABLES REPORT")
        print("=" * 70)
        
        total_unused = len(unused_vars['member']) + len(unused_vars['local'])
        
        if total_unused == 0:
            print("\n✓ No unused variables found!")
            print("  All variables are being used.")
        else:
            # Member variables
            if unused_vars['member']:
                print(f"\n⚠ Found {len(unused_vars['member'])} unused member variable(s):\n")
                
                current_class = None
                for var in sorted(unused_vars['member'], key=lambda x: (x['class'], x['line'])):
                    if var['class'] != current_class:
                        current_class = var['class']
                        print(f"\n  Class: {current_class}")
                    
                    print(f"    • {var['name']}")
                    print(f"      Line: {var['line']}")
            
            # Local variables
            if unused_vars['local']:
                print(f"\n⚠ Found {len(unused_vars['local'])} unused local variable(s):\n")
                
                current_func = None
                for var in sorted(unused_vars['local'], key=lambda x: (x['function'], x['line'])):
                    if var['function'] != current_func:
                        current_func = var['function']
                        print(f"\n  Function: {current_func}()")
                    
                    print(f"    • {var['name']}")
                    print(f"      Line: {var['line']}")
            
            print(f"\n  Total: {total_unused} unused variable(s)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze C++ files for unused functions and variables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s mycode.cpp --functions        # Find unused functions
  %(prog)s mycode.cpp --variables        # Find unused variables
  %(prog)s mycode.cpp --all              # Find both
  %(prog)s mycode.cpp -f -v              # Short form
        '''
    )
    
    parser.add_argument('cpp_file', help='C++ source file to analyze')
    parser.add_argument('-f', '--functions', action='store_true',
                        help='Find unused functions')
    parser.add_argument('-v', '--variables', action='store_true',
                        help='Find unused variables')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Find both unused functions and variables')
    
    args = parser.parse_args()
    
    # Validate file exists
    if not Path(args.cpp_file).exists():
        print(f"Error: File '{args.cpp_file}' does not exist.")
        sys.exit(1)
    
    # If no flags specified, show help
    if not (args.functions or args.variables or args.all):
        parser.print_help()
        print("\n⚠ Error: Please specify at least one analysis type (-f, -v, or -a)")
        sys.exit(1)
    
    # If --all is specified, enable both
    if args.all:
        args.functions = True
        args.variables = True
    
    # Create analyzer
    print(f"Analyzing {args.cpp_file}...")
    analyzer = CppUnusedAnalyzer(args.cpp_file)
    
    # Run requested analyses
    if args.functions:
        unused_functions = analyzer.find_unused_functions()
        analyzer.print_unused_functions(unused_functions)
    
    if args.variables:
        unused_variables = analyzer.find_unused_variables()
        analyzer.print_unused_variables(unused_variables)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
