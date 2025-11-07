/*
 * C++ Class Extractor
 * Extracts class definitions and related functions from C++ source files
 * 
 * Usage: ./extract_class <input.cpp> <class_name> <output.cpp>
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE_LENGTH 4096
#define MAX_CLASS_NAME 256
#define MAX_HEADERS 100

typedef struct {
    char *content;
    size_t size;
    size_t capacity;
} Buffer;

typedef struct {
    char *headers[MAX_HEADERS];
    size_t count;
} HeaderList;

// Initialize a dynamic buffer
void buffer_init(Buffer *buf) {
    buf->capacity = 8192;
    buf->size = 0;
    buf->content = (char *)malloc(buf->capacity);
    if (!buf->content) {
        fprintf(stderr, "Error: Failed to allocate buffer\n");
        exit(1);
    }
    buf->content[0] = '\0';
}

// Append string to buffer
void buffer_append(Buffer *buf, const char *str) {
    if (!buf || !str || !buf->content) return;
    
    size_t len = strlen(str);
    while (buf->size + len + 1 > buf->capacity) {
        buf->capacity *= 2;
        char *new_content = (char *)realloc(buf->content, buf->capacity);
        if (!new_content) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            return;
        }
        buf->content = new_content;
    }
    strcpy(buf->content + buf->size, str);
    buf->size += len;
}

// Free buffer
void buffer_free(Buffer *buf) {
    if (buf && buf->content) {
        free(buf->content);
        buf->content = NULL;
    }
    buf->size = 0;
    buf->capacity = 0;
}

// Initialize header list
void header_list_init(HeaderList *list) {
    list->count = 0;
    for (int i = 0; i < MAX_HEADERS; i++) {
        list->headers[i] = NULL;
    }
}

// Add header to list (if not already present)
void header_list_add(HeaderList *list, const char *header) {
    if (list->count >= MAX_HEADERS) return;
    
    // Check if already exists
    for (size_t i = 0; i < list->count; i++) {
        if (list->headers[i] && strcmp(list->headers[i], header) == 0) {
            return; // Already in list
        }
    }
    
    // Add new header
    list->headers[list->count] = strdup(header);
    if (list->headers[list->count]) {
        list->count++;
    }
}

// Free header list
void header_list_free(HeaderList *list) {
    for (size_t i = 0; i < list->count; i++) {
        if (list->headers[i]) {
            free(list->headers[i]);
            list->headers[i] = NULL;
        }
    }
    list->count = 0;
}

// Trim whitespace from beginning of string
char *trim_left(char *str) {
    while (isspace((unsigned char)*str)) str++;
    return str;
}

// Check if string contains a word (as a whole word)
int contains_word(const char *str, const char *word) {
    const char *pos = str;
    size_t word_len = strlen(word);
    
    while ((pos = strstr(pos, word)) != NULL) {
        // Check if it's a whole word (not part of another identifier)
        int before_ok = (pos == str) || (!isalnum((unsigned char)pos[-1]) && pos[-1] != '_');
        int after_ok = !isalnum((unsigned char)pos[word_len]) && pos[word_len] != '_';
        
        if (before_ok && after_ok) {
            return 1;
        }
        pos++;
    }
    return 0;
}

// Check if line is a class definition
int is_class_definition(const char *line, const char *class_name) {
    char *trimmed = trim_left((char *)line);
    
    // Check for "class ClassName" or "struct ClassName"
    if (strncmp(trimmed, "class ", 6) == 0 || strncmp(trimmed, "struct ", 7) == 0) {
        return contains_word(trimmed, class_name);
    }
    return 0;
}

// Check if line is a method definition (ClassName::method)
int is_method_definition(const char *line, const char *class_name) {
    char pattern[MAX_CLASS_NAME + 10];
    snprintf(pattern, sizeof(pattern), "%s::", class_name);
    return strstr(line, pattern) != NULL;
}

// Check if line is an include directive
int is_include_directive(const char *line) {
    char *trimmed = trim_left((char *)line);
    return strncmp(trimmed, "#include", 8) == 0;
}

// Count braces to track nesting level
int count_braces(const char *line, int *open, int *close) {
    *open = 0;
    *close = 0;
    int in_string = 0;
    int in_char = 0;
    int in_comment = 0;
    
    for (const char *p = line; *p; p++) {
        // Handle escape sequences
        if (*p == '\\' && (in_string || in_char)) {
            p++;
            continue;
        }
        
        // Track string literals
        if (*p == '"' && !in_char && !in_comment) {
            in_string = !in_string;
            continue;
        }
        
        // Track character literals
        if (*p == '\'' && !in_string && !in_comment) {
            in_char = !in_char;
            continue;
        }
        
        // Track comments
        if (!in_string && !in_char) {
            if (p[0] == '/' && p[1] == '/') {
                break; // Rest of line is comment
            }
            if (p[0] == '/' && p[1] == '*') {
                in_comment = 1;
                p++;
                continue;
            }
            if (in_comment && p[0] == '*' && p[1] == '/') {
                in_comment = 0;
                p++;
                continue;
            }
        }
        
        // Count braces outside strings and comments
        if (!in_string && !in_char && !in_comment) {
            if (*p == '{') (*open)++;
            if (*p == '}') (*close)++;
        }
    }
    
    return *open - *close;
}

// Analyze code and determine necessary headers based on types used
void analyze_code_for_headers(const char *code, HeaderList *headers) {
    // Common type to header mappings
    struct TypeHeaderMap {
        const char *type;
        const char *header;
    } type_map[] = {
        {"std::string", "#include <string>"},
        {"std::vector", "#include <vector>"},
        {"std::map", "#include <map>"},
        {"std::set", "#include <set>"},
        {"std::list", "#include <list>"},
        {"std::queue", "#include <queue>"},
        {"std::stack", "#include <stack>"},
        {"std::deque", "#include <deque>"},
        {"std::array", "#include <array>"},
        {"std::pair", "#include <utility>"},
        {"std::shared_ptr", "#include <memory>"},
        {"std::unique_ptr", "#include <memory>"},
        {"std::weak_ptr", "#include <memory>"},
        {"std::cout", "#include <iostream>"},
        {"std::cin", "#include <iostream>"},
        {"std::endl", "#include <iostream>"},
        {"std::ifstream", "#include <fstream>"},
        {"std::ofstream", "#include <fstream>"},
        {"std::stringstream", "#include <sstream>"},
        {"std::ostringstream", "#include <sstream>"},
        {"std::istringstream", "#include <sstream>"},
        {"std::function", "#include <functional>"},
        {"std::thread", "#include <thread>"},
        {"std::mutex", "#include <mutex>"},
        {"std::exception", "#include <exception>"},
        {"std::runtime_error", "#include <stdexcept>"},
        {NULL, NULL}  // Sentinel
    };
    
    // Check for each type in the code
    for (int i = 0; type_map[i].type != NULL; i++) {
        if (strstr(code, type_map[i].type) != NULL) {
            header_list_add(headers, type_map[i].header);
        }
    }
}

// Extract class and related functions
int extract_class(const char *input_file, const char *class_name, const char *output_file) {
    FILE *fin = fopen(input_file, "r");
    if (!fin) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", input_file);
        return 1;
    }
    
    FILE *fout = fopen(output_file, "w");
    if (!fout) {
        fprintf(stderr, "Error: Cannot open output file '%s'\n", output_file);
        fclose(fin);
        return 1;
    }
    
    char line[MAX_LINE_LENGTH];
    Buffer current_block;
    Buffer all_extracted_code;
    HeaderList source_headers;
    HeaderList inferred_headers;
    
    buffer_init(&current_block);
    buffer_init(&all_extracted_code);
    header_list_init(&source_headers);
    header_list_init(&inferred_headers);
    
    int in_class = 0;
    int in_method = 0;
    int brace_level = 0;
    int found_anything = 0;
    
    // First pass: collect all includes from source file and extract class code
    while (fgets(line, sizeof(line), fin)) {
        // Collect include directives
        if (is_include_directive(line)) {
            header_list_add(&source_headers, line);
        }
        
        int open_braces, close_braces;
        int brace_delta = count_braces(line, &open_braces, &close_braces);
        
        // Check if starting a class definition
        if (!in_class && !in_method && is_class_definition(line, class_name)) {
            in_class = 1;
            brace_level = 0;
            buffer_append(&current_block, line);
            brace_level += brace_delta;
            found_anything = 1;
            continue;
        }
        
        // Check if starting a method definition
        if (!in_class && !in_method && is_method_definition(line, class_name)) {
            in_method = 1;
            brace_level = 0;
            buffer_append(&current_block, line);
            brace_level += brace_delta;
            found_anything = 1;
            continue;
        }
        
        // Inside class definition
        if (in_class) {
            buffer_append(&current_block, line);
            brace_level += brace_delta;
            
            if (brace_level <= 0) {
                // End of class definition
                buffer_append(&all_extracted_code, current_block.content);
                buffer_append(&all_extracted_code, "\n");
                buffer_free(&current_block);
                buffer_init(&current_block);
                in_class = 0;
                brace_level = 0;
            }
            continue;
        }
        
        // Inside method definition
        if (in_method) {
            buffer_append(&current_block, line);
            brace_level += brace_delta;
            
            if (brace_level <= 0) {
                // End of method definition
                buffer_append(&all_extracted_code, current_block.content);
                buffer_append(&all_extracted_code, "\n");
                buffer_free(&current_block);
                buffer_init(&current_block);
                in_method = 0;
                brace_level = 0;
            }
            continue;
        }
    }
    
    if (!found_anything) {
        fprintf(stderr, "Warning: No class or methods found for '%s'\n", class_name);
        buffer_free(&current_block);
        buffer_free(&all_extracted_code);
        header_list_free(&source_headers);
        header_list_free(&inferred_headers);
        fclose(fin);
        fclose(fout);
        return 2;
    }
    
    // Analyze extracted code to infer necessary headers
    analyze_code_for_headers(all_extracted_code.content, &inferred_headers);
    
    // Save counts
    size_t inferred_count = inferred_headers.count;
    size_t source_count = source_headers.count;
    
    // Write output file
    fprintf(fout, "// Extracted from: %s\n", input_file);
    fprintf(fout, "// Class: %s\n\n", class_name);
    
    // Write inferred headers first (common standard library headers)
    if (inferred_count > 0) {
        fprintf(fout, "// Inferred necessary headers:\n");
        for (size_t i = 0; i < inferred_count; i++) {
            if (inferred_headers.headers[i]) {
                fprintf(fout, "%s\n", inferred_headers.headers[i]);
            }
        }
        fprintf(fout, "\n");
    }
    
    // Write headers that were in the source file
    if (source_count > 0) {
        fprintf(fout, "// Headers from source file:\n");
        for (size_t i = 0; i < source_count; i++) {
            if (source_headers.headers[i]) {
                // Check if not a duplicate of inferred headers
                int is_duplicate = 0;
                for (size_t j = 0; j < inferred_count; j++) {
                    if (inferred_headers.headers[j] && 
                        strstr(source_headers.headers[i], inferred_headers.headers[j] + 9)) {
                        // +9 skips "#include "
                        is_duplicate = 1;
                        break;
                    }
                }
                if (!is_duplicate) {
                    fprintf(fout, "%s", source_headers.headers[i]);
                }
            }
        }
        fprintf(fout, "\n");
    }
    
    // Write the extracted code
    if (all_extracted_code.content) {
        fprintf(fout, "%s", all_extracted_code.content);
    }
    
    buffer_free(&current_block);
    buffer_free(&all_extracted_code);
    header_list_free(&source_headers);
    header_list_free(&inferred_headers);
    fclose(fin);
    fclose(fout);
    
    printf("Successfully extracted class '%s' to '%s'\n", class_name, output_file);
    printf("  - Found %zu inferred header(s)\n", inferred_count);
    printf("  - Found %zu source header(s)\n", source_count);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.cpp> <class_name> <output.cpp>\n", argv[0]);
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s source.cpp MyClass output.cpp\n", argv[0]);
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *class_name = argv[2];
    const char *output_file = argv[3];
    
    if (strlen(class_name) >= MAX_CLASS_NAME) {
        fprintf(stderr, "Error: Class name too long (max %d characters)\n", MAX_CLASS_NAME - 1);
        return 1;
    }
    
    return extract_class(input_file, class_name, output_file);
}
