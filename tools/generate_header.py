#!/usr/bin/env python3
"""
C++ Header File Generator

This script analyzes a C++ source file (.cpp) and automatically generates
a corresponding header file (.h) with:
- Include guards
- Class declarations
- Function declarations
- Necessary includes
- Proper formatting
"""

import re
import sys
from pathlib import Path
from typing import List, Set, Tuple, Optional, Dict


class CppHeaderGenerator:
    def __init__(self, cpp_filepath: str):
        self.cpp_filepath = cpp_filepath
        self.cpp_filename = Path(cpp_filepath).name
        self.header_filename = self._get_header_filename()
        self.content = self._read_file()
        self.content_no_comments = self._remove_comments(self.content)
        
    def _read_file(self) -> str:
        """Read the C++ file content."""
        try:
            with open(self.cpp_filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: File '{self.cpp_filepath}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    
    def _get_header_filename(self) -> str:
        """Generate header filename from cpp filename."""
        cpp_path = Path(self.cpp_filepath)
        return cpp_path.stem + '.h'
    
    def _remove_comments(self, text: str) -> str:
        """Remove C++ comments from text."""
        # Remove single-line comments
        text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
        # Remove multi-line comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text
    
    def _generate_include_guard(self) -> str:
        """Generate include guard macro name."""
        # Convert filename to uppercase and replace special chars with underscore
        guard = self.header_filename.upper().replace('.', '_').replace('-', '_')
        return guard
    
    def _extract_includes(self) -> List[str]:
        """Extract #include directives from the source file."""
        includes = []
        include_pattern = r'^\s*#include\s+[<"]([^>"]+)[>"]'
        
        for line in self.content.split('\n'):
            match = re.match(include_pattern, line)
            if match:
                include_file = match.group(1)
                # Skip including the generated header itself
                if include_file != self.header_filename:
                    includes.append(line.strip())
        
        return includes
    
    def _extract_classes(self) -> List[Dict]:
        """Extract class declarations."""
        classes = []
        
        # Pattern to match class definitions
        class_pattern = r'\bclass\s+(\w+)(?:\s*:\s*([^{]+))?\s*{'
        
        for match in re.finditer(class_pattern, self.content_no_comments):
            class_name = match.group(1)
            inheritance = match.group(2).strip() if match.group(2) else None
            
            # Extract the full class body
            class_body = self._extract_class_body(match.start(), self.content_no_comments)
            
            # Extract class members
            members = self._extract_class_members(class_body)
            
            classes.append({
                'name': class_name,
                'inheritance': inheritance,
                'members': members,
                'full_declaration': self._format_class_declaration(class_name, inheritance, members)
            })
        
        return classes
    
    def _extract_class_body(self, start_pos: int, text: str) -> str:
        """Extract the complete body of a class."""
        # Find the opening brace
        i = start_pos
        while i < len(text) and text[i] != '{':
            i += 1
        
        if i >= len(text):
            return ""
        
        brace_count = 0
        start = i
        
        while i < len(text):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i+1]
            i += 1
        
        return ""
    
    def _remove_function_bodies(self, class_body: str) -> str:
        """Remove function bodies, keeping only declarations."""
        # This function replaces inline function definitions with just declarations
        result = []
        lines = class_body.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this line starts a function definition (has opening brace)
            if '{' in line and not line.startswith('class'):
                # Find the function signature part before the brace
                sig_end = line.find('{')
                signature = line[:sig_end].strip()
                
                # If it's a function declaration, replace with semicolon
                if '(' in signature and ')' in signature:
                    result.append(signature + ';')
                    
                    # Skip lines until we find the matching closing brace
                    brace_count = line.count('{') - line.count('}')
                    i += 1
                    while i < len(lines) and brace_count > 0:
                        brace_count += lines[i].count('{') - lines[i].count('}')
                        i += 1
                    continue
            
            result.append(lines[i])
            i += 1
        
        return '\n'.join(result)
    
    def _extract_class_members(self, class_body: str) -> Dict:
        """Extract public, private, and protected members from class body."""
        members = {
            'public': [],
            'private': [],
            'protected': []
        }
        
        # Split by access specifiers
        current_access = 'private'  # Default for class
        
        # Remove function bodies by replacing them with semicolons
        # This prevents extracting lines from within function implementations
        cleaned_body = self._remove_function_bodies(class_body)
        
        lines = cleaned_body.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check for access specifier
            if re.match(r'^\s*(public|private|protected)\s*:\s*$', line):
                access_match = re.match(r'^\s*(public|private|protected)\s*:', line)
                if access_match:
                    current_access = access_match.group(1)
                continue
            
            # Skip empty lines, braces, and class keyword
            if not line or line in ['{', '}'] or line.startswith('class '):
                continue
            
            # Check for constructor/destructor pattern (no return type)
            ctor_dtor_pattern = r'^(?:explicit\s+)?~?(\w+)\s*\(([^)]*)\)\s*(?::\s*[^;{]+)?;'
            ctor_match = re.match(ctor_dtor_pattern, line)
            
            if ctor_match:
                name = ctor_match.group(1)
                params = ctor_match.group(2).strip()
                is_explicit = 'explicit' in line
                explicit_str = 'explicit ' if is_explicit else ''
                
                # It's a constructor if name matches class name or destructor if starts with ~
                if name.startswith('~') or '~' in line:
                    members[current_access].append(line if ';' in line else line + ';')
                else:
                    members[current_access].append(f"{explicit_str}{name}({params});")
                continue
            
            # Extract member declarations (functions and variables)
            # Function pattern - must end with semicolon
            func_pattern = r'^(?:virtual\s+|static\s+|inline\s+|explicit\s+)*(\w+(?:\s*[*&]+)?(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?(?:=\s*0\s*)?;'
            func_match = re.match(func_pattern, line)
            
            if func_match:
                return_type = func_match.group(1).strip()
                func_name = func_match.group(2).strip()
                params = func_match.group(3).strip()
                
                # Check if it's a pure virtual function
                is_pure_virtual = '= 0' in line
                is_virtual = 'virtual' in line or is_pure_virtual
                is_const = 'const' in line
                is_static = 'static' in line
                
                # Skip constructors and destructors in some contexts
                modifiers = []
                if is_virtual:
                    modifiers.append('virtual')
                if is_static:
                    modifiers.append('static')
                
                modifier_str = ' '.join(modifiers) + ' ' if modifiers else ''
                const_str = ' const' if is_const else ''
                pure_virtual_str = ' = 0' if is_pure_virtual else ''
                
                declaration = f"{modifier_str}{return_type} {func_name}({params}){const_str}{pure_virtual_str};"
                members[current_access].append(declaration)
            else:
                # Variable pattern - must end with semicolon
                var_pattern = r'^(?:static\s+|const\s+|mutable\s+)*(\w+(?:\s*[*&]+)?(?:<[^>]+>)?)\s+(\w+)\s*;'
                var_match = re.match(var_pattern, line)
                
                if var_match:
                    var_type = var_match.group(1).strip()
                    var_name = var_match.group(2).strip()
                    is_static = 'static' in line
                    is_const = 'const' in line
                    is_mutable = 'mutable' in line
                    
                    modifiers = []
                    if is_static:
                        modifiers.append('static')
                    if is_const:
                        modifiers.append('const')
                    if is_mutable:
                        modifiers.append('mutable')
                    
                    modifier_str = ' '.join(modifiers) + ' ' if modifiers else ''
                    members[current_access].append(f"{modifier_str}{var_type} {var_name};")
        
        return members
    
    def _format_class_declaration(self, class_name: str, inheritance: Optional[str], members: Dict) -> str:
        """Format a complete class declaration for the header."""
        lines = []
        
        # Class declaration with inheritance
        if inheritance:
            lines.append(f"class {class_name} : {inheritance} {{")
        else:
            lines.append(f"class {class_name} {{")
        
        # Add members by access level
        for access in ['public', 'protected', 'private']:
            if members[access]:
                lines.append(f"{access}:")
                for member in members[access]:
                    lines.append(f"    {member}")
                lines.append("")
        
        lines.append("};")
        
        return '\n'.join(lines)
    
    def _extract_standalone_functions(self) -> List[str]:
        """Extract standalone function declarations (not in classes)."""
        functions = []
        
        # First get all class member function names to exclude them
        classes = self._extract_classes()
        class_method_names = set()
        for cls in classes:
            for access in ['public', 'private', 'protected']:
                for member in cls['members'][access]:
                    # Extract function name from declaration
                    func_match = re.match(r'(?:virtual\s+|static\s+)?[\w<>*&]+\s+(\w+)\s*\(', member)
                    if func_match:
                        class_method_names.add(func_match.group(1))
        
        # Remove class bodies to avoid extracting class methods
        text_no_classes = self._remove_class_bodies(self.content_no_comments)
        
        # Pattern for function definitions
        func_pattern = r'^[\s]*(?:inline\s+|static\s+|extern\s+)?([\w<>*&]+)\s+(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?{'
        
        for match in re.finditer(func_pattern, text_no_classes, re.MULTILINE):
            return_type = match.group(1).strip()
            func_name = match.group(2).strip()
            params = match.group(3).strip()
            
            # Skip main function
            if func_name == 'main':
                continue
            
            # Skip if this is a class method
            if func_name in class_method_names:
                continue
            
            # Check for inline keyword
            is_inline = 'inline' in self.content[max(0, match.start()-20):match.start()]
            inline_str = 'inline ' if is_inline else ''
            
            declaration = f"{inline_str}{return_type} {func_name}({params});"
            functions.append(declaration)
        
        return functions
    
    def _remove_class_bodies(self, text: str) -> str:
        """Remove class bodies to isolate standalone functions."""
        # Replace class bodies with empty space
        class_pattern = r'\bclass\s+\w+[^{]*{[^}]*}'
        
        # This is a simple approach - for nested braces, we need a more sophisticated method
        result = text
        while True:
            new_result = re.sub(r'\bclass\s+\w+[^{]*\{(?:[^{}]|\{[^{}]*\})*\}', '', result)
            if new_result == result:
                break
            result = new_result
        
        return result
    
    def _extract_structs(self) -> List[str]:
        """Extract struct declarations."""
        structs = []
        
        # Pattern to match struct definitions
        struct_pattern = r'\bstruct\s+(\w+)\s*{([^}]*)};'
        
        for match in re.finditer(struct_pattern, self.content_no_comments, re.DOTALL):
            struct_name = match.group(1)
            struct_body = match.group(2).strip()
            
            # Format the struct declaration
            declaration = f"struct {struct_name} {{\n"
            for line in struct_body.split('\n'):
                line = line.strip()
                if line:
                    declaration += f"    {line}\n"
            declaration += "};"
            
            structs.append(declaration)
        
        return structs
    
    def _extract_enums(self) -> List[str]:
        """Extract enum declarations."""
        enums = []
        
        # Pattern to match enum definitions
        enum_pattern = r'\benum(?:\s+class)?\s+(\w+)\s*{([^}]*)};'
        
        for match in re.finditer(enum_pattern, self.content_no_comments, re.DOTALL):
            enums.append(match.group(0))
        
        return enums
    
    def _extract_typedefs(self) -> List[str]:
        """Extract typedef and using declarations."""
        typedefs = []
        
        # Pattern for typedef
        typedef_pattern = r'^\s*typedef\s+[^;]+;'
        # Pattern for using
        using_pattern = r'^\s*using\s+\w+\s*=\s*[^;]+;'
        
        for line in self.content_no_comments.split('\n'):
            if re.match(typedef_pattern, line) or re.match(using_pattern, line):
                typedefs.append(line.strip())
        
        return typedefs
    
    def _extract_namespace(self) -> Optional[str]:
        """Extract namespace if the code is wrapped in one."""
        namespace_pattern = r'^\s*namespace\s+(\w+)\s*{'
        
        for line in self.content_no_comments.split('\n'):
            match = re.match(namespace_pattern, line)
            if match:
                return match.group(1)
        
        return None
    
    def generate_header(self) -> str:
        """Generate the complete header file content."""
        lines = []
        
        # Include guard start
        guard = self._generate_include_guard()
        lines.append(f"#ifndef {guard}")
        lines.append(f"#define {guard}")
        lines.append("")
        
        # File comment
        lines.append(f"// {self.header_filename}")
        lines.append(f"// Generated from {self.cpp_filename}")
        lines.append("")
        
        # Includes
        includes = self._extract_includes()
        if includes:
            for include in includes:
                lines.append(include)
            lines.append("")
        
        # Namespace start
        namespace = self._extract_namespace()
        if namespace:
            lines.append(f"namespace {namespace} {{")
            lines.append("")
        
        # Forward declarations (optional, could be enhanced)
        lines.append("// Forward declarations")
        lines.append("")
        
        # Enums
        enums = self._extract_enums()
        if enums:
            lines.append("// Enumerations")
            for enum in enums:
                lines.append(enum)
                lines.append("")
        
        # Typedefs
        typedefs = self._extract_typedefs()
        if typedefs:
            lines.append("// Type definitions")
            for typedef in typedefs:
                lines.append(typedef)
            lines.append("")
        
        # Structs
        structs = self._extract_structs()
        if structs:
            lines.append("// Structures")
            for struct in structs:
                lines.append(struct)
                lines.append("")
        
        # Classes
        classes = self._extract_classes()
        if classes:
            lines.append("// Classes")
            for cls in classes:
                lines.append(cls['full_declaration'])
                lines.append("")
        
        # Standalone functions
        functions = self._extract_standalone_functions()
        if functions:
            lines.append("// Function declarations")
            for func in functions:
                lines.append(func)
            lines.append("")
        
        # Namespace end
        if namespace:
            lines.append(f"}} // namespace {namespace}")
            lines.append("")
        
        # Include guard end
        lines.append(f"#endif // {guard}")
        
        return '\n'.join(lines)
    
    def save_header(self, output_path: Optional[str] = None):
        """Save the generated header to a file."""
        if output_path is None:
            output_path = self.header_filename
        
        header_content = self.generate_header()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(header_content)
            print(f"✓ Header file generated successfully: {output_path}")
        except Exception as e:
            print(f"Error writing header file: {e}")
            sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_header.py <cpp_file> [output_header]")
        print("\nGenerates a header file from a C++ source file.")
        print("\nExamples:")
        print("  python generate_header.py mycode.cpp")
        print("  python generate_header.py mycode.cpp custom_header.h")
        sys.exit(1)
    
    cpp_file = sys.argv[1]
    output_header = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(cpp_file).exists():
        print(f"Error: File '{cpp_file}' does not exist.")
        sys.exit(1)
    
    print(f"Analyzing {cpp_file}...")
    generator = CppHeaderGenerator(cpp_file)
    
    print("Generating header file...")
    generator.save_header(output_header)
    
    print("\nGenerated header structure:")
    print("  - Include guards")
    print("  - Required includes")
    if generator._extract_classes():
        print(f"  - {len(generator._extract_classes())} class(es)")
    if generator._extract_standalone_functions():
        print(f"  - {len(generator._extract_standalone_functions())} function(s)")
    if generator._extract_structs():
        print(f"  - {len(generator._extract_structs())} struct(s)")
    if generator._extract_enums():
        print(f"  - {len(generator._extract_enums())} enum(s)")


if __name__ == "__main__":
    main()