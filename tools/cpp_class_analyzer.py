#!/usr/bin/env python3
"""
C++ Class Dependency Analyzer

This script analyzes a C++ file and lists all classes that are used by a specified class.
It detects class usage through:
- Member variables
- Method parameters
- Method return types
- Inheritance (base classes)
- Template parameters
"""

import re
import sys
from typing import Set, List, Optional
from pathlib import Path


class CppClassAnalyzer:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.content = self._read_file()
        self.classes = self._extract_all_classes()
    
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
    
    def _extract_all_classes(self) -> Set[str]:
        """Extract all class names defined in the file."""
        class_pattern = r'\bclass\s+(\w+)'
        struct_pattern = r'\bstruct\s+(\w+)'
        
        classes = set()
        classes.update(re.findall(class_pattern, self.content))
        classes.update(re.findall(struct_pattern, self.content))
        
        return classes
    
    def _remove_comments(self, text: str) -> str:
        """Remove C++ comments from text."""
        # Remove single-line comments
        text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
        # Remove multi-line comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text
    
    def _extract_class_body(self, class_name: str) -> Optional[str]:
        """Extract the body of a specific class."""
        # Remove comments first
        content = self._remove_comments(self.content)
        
        # Pattern to match class/struct definition
        pattern = rf'\b(?:class|struct)\s+{re.escape(class_name)}\s*(?::\s*[^{{]*)?{{'
        
        match = re.search(pattern, content)
        if not match:
            return None
        
        start = match.end() - 1  # Position at the opening brace
        brace_count = 0
        i = start
        
        # Find matching closing brace
        while i < len(content):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Include the base class part
                    base_start = match.start()
                    return content[base_start:i+1]
            i += 1
        
        return None
    
    def _extract_base_classes(self, class_body: str, class_name: str) -> Set[str]:
        """Extract base classes from inheritance."""
        base_classes = set()
        
        # Pattern for inheritance: class A : public B, private C { ... }
        pattern = rf'\b(?:class|struct)\s+{re.escape(class_name)}\s*:\s*([^{{]+){{'
        match = re.search(pattern, class_body)
        
        if match:
            inheritance_list = match.group(1)
            # Extract base class names (after public/private/protected keywords)
            base_pattern = r'(?:public|private|protected)\s+(\w+(?:<[^>]+>)?)'
            bases = re.findall(base_pattern, inheritance_list)
            base_classes.update(bases)
        
        return base_classes
    
    def _extract_type_names(self, text: str) -> Set[str]:
        """Extract potential class/type names from declarations."""
        types = set()
        
        # Match type declarations (remove common C++ keywords)
        type_pattern = r'\b([A-Z]\w*(?:<[^>]+>)?)\b'
        potential_types = re.findall(type_pattern, text)
        
        # C++ keywords and primitives to exclude
        cpp_keywords = {
            'auto', 'break', 'case', 'catch', 'class', 'const', 'continue',
            'default', 'delete', 'do', 'else', 'enum', 'explicit', 'export',
            'extern', 'false', 'for', 'friend', 'goto', 'if', 'inline',
            'mutable', 'namespace', 'new', 'nullptr', 'operator', 'private',
            'protected', 'public', 'register', 'return', 'sizeof', 'static',
            'struct', 'switch', 'template', 'this', 'throw', 'true', 'try',
            'typedef', 'typeid', 'typename', 'union', 'using', 'virtual',
            'void', 'volatile', 'while', 'bool', 'char', 'int', 'float',
            'double', 'long', 'short', 'signed', 'unsigned', 'wchar_t'
        }
        
        for t in potential_types:
            # Extract base type from templates
            base_type = re.match(r'(\w+)', t)
            if base_type:
                type_name = base_type.group(1)
                if type_name not in cpp_keywords and type_name[0].isupper():
                    types.add(type_name)
        
        return types
    
    def _extract_member_variables(self, class_body: str) -> Set[str]:
        """Extract types of member variables."""
        types = set()
        
        # Pattern for member variable declarations
        # Matches: Type varName; or Type* varName; or Type& varName;
        member_pattern = r'\b(\w+(?:<[^>]+>)?)\s*[*&]?\s+\w+\s*[;=]'
        matches = re.findall(member_pattern, class_body)
        
        types.update(self._extract_type_names(' '.join(matches)))
        
        # Also check for smart pointers and containers
        smart_ptr_pattern = r'(?:std::)?(?:unique_ptr|shared_ptr|weak_ptr|vector|list|map|set|unordered_map|unordered_set)<\s*(\w+)'
        smart_matches = re.findall(smart_ptr_pattern, class_body)
        types.update(smart_matches)
        
        return types
    
    def _extract_method_types(self, class_body: str) -> Set[str]:
        """Extract types from method parameters and return types."""
        types = set()
        
        # Pattern for method declarations
        method_pattern = r'\b(\w+(?:<[^>]+>)?)\s+\w+\s*\([^)]*\)'
        matches = re.findall(method_pattern, class_body)
        types.update(self._extract_type_names(' '.join(matches)))
        
        # Extract parameter types
        param_pattern = r'\(([^)]*)\)'
        param_matches = re.findall(param_pattern, class_body)
        for params in param_matches:
            types.update(self._extract_type_names(params))
        
        return types
    
    def find_used_classes(self, class_name: str) -> Set[str]:
        """Find all classes used by the specified class."""
        # Check if the class exists
        if class_name not in self.classes:
            print(f"Warning: Class '{class_name}' not found in the file.")
            return set()
        
        class_body = self._extract_class_body(class_name)
        if not class_body:
            print(f"Error: Could not extract body of class '{class_name}'.")
            return set()
        
        used_classes = set()
        
        # Extract base classes
        used_classes.update(self._extract_base_classes(class_body, class_name))
        
        # Extract types from member variables
        used_classes.update(self._extract_member_variables(class_body))
        
        # Extract types from methods
        used_classes.update(self._extract_method_types(class_body))
        
        # Filter to only include classes that are defined in the file or look like classes
        # Remove the class itself
        used_classes.discard(class_name)
        
        return used_classes


def main():
    if len(sys.argv) < 3:
        print("Usage: python cpp_class_analyzer.py <cpp_file> <class_name>")
        print("\nExample:")
        print("  python cpp_class_analyzer.py example.cpp MyClass")
        sys.exit(1)
    
    cpp_file = sys.argv[1]
    class_name = sys.argv[2]
    
    if not Path(cpp_file).exists():
        print(f"Error: File '{cpp_file}' does not exist.")
        sys.exit(1)
    
    analyzer = CppClassAnalyzer(cpp_file)
    used_classes = analyzer.find_used_classes(class_name)
    
    if used_classes:
        print(f"\nClasses used by '{class_name}':")
        print("-" * 40)
        for cls in sorted(used_classes):
            print(f"  - {cls}")
        print(f"\nTotal: {len(used_classes)} class(es)")
    else:
        print(f"\nNo classes found to be used by '{class_name}'.")


if __name__ == "__main__":
    main()
