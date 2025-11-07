#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE 1024
#define MAX_ITEMS 1000

typedef struct {
    char text[MAX_LINE];
    int is_completed;
} TodoItem;

typedef struct {
    char header[MAX_LINE];
    TodoItem items[MAX_ITEMS];
    int count;
} Section;

// Trim leading/trailing whitespace
char* trim(char* str) {
    char* end;
    while (isspace((unsigned char)*str)) str++;
    if (*str == 0) return str;
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return str;
}

// Check if line is a completed task
int is_completed_task(const char* line) {
    const char* trimmed = line;
    while (isspace((unsigned char)*trimmed)) trimmed++;
    return (strncmp(trimmed, "- [x]", 5) == 0 || 
            strncmp(trimmed, "- [X]", 5) == 0);
}

// Check if line is an incomplete task
int is_incomplete_task(const char* line) {
    const char* trimmed = line;
    while (isspace((unsigned char)*trimmed)) trimmed++;
    return (strncmp(trimmed, "- []", 4) == 0 || 
            strncmp(trimmed, "- [ ]", 5) == 0);
}

int main(int argc, char* argv[]) {
    const char* input_file = "todo.md";
    const char* temp_file = "todo.md.tmp";

    FILE* input = fopen(input_file, "r");
    if (!input) {
        perror("Error opening todo.md");
        fprintf(stderr, "Make sure todo.md exists in the current directory\n");
        return 1;
    }

    FILE* output = fopen(temp_file, "w");
    if (!output) {
        perror("Error creating temporary file");
        fclose(input);
        return 1;
    }

    char line[MAX_LINE];
    Section software_todo = {0};
    Section software_completed = {0};
    Section analysis_todo = {0};
    Section analysis_completed = {0};
    
    strcpy(software_todo.header, "## to do software goals\n");
    strcpy(software_completed.header, "## completed software goals\n");
    strcpy(analysis_todo.header, "## to do analysis goals\n");
    strcpy(analysis_completed.header, "## completed analysis goals\n");

    Section* current_section = NULL;
    int in_preamble = 1;
    char preamble[10000] = "";
    char middle_content[10000] = "";
    
    while (fgets(line, sizeof(line), input)) {
        // Check for section headers
        if (strstr(line, "## to do software goals")) {
            current_section = &software_todo;
            in_preamble = 0;
            continue;
        }
        else if (strstr(line, "## completed software goals")) {
            current_section = &software_completed;
            continue;
        }
        else if (strstr(line, "## to do analysis goals")) {
            current_section = &analysis_todo;
            continue;
        }
        else if (strstr(line, "## completed analysis goals")) {
            current_section = &analysis_completed;
            continue;
        }
        else if (strstr(line, "## input files") || 
                 strstr(line, "## output files")) {
            // Store middle sections
            strcat(middle_content, line);
            current_section = NULL;
            continue;
        }
        
        // Handle content
        if (in_preamble) {
            strcat(preamble, line);
        }
        else if (current_section == NULL) {
            // Middle content (input/output files sections)
            strcat(middle_content, line);
        }
        else if (is_completed_task(line) || is_incomplete_task(line)) {
            // Add task to current section
            if (current_section->count < MAX_ITEMS) {
                size_t len = strlen(line);
                if (len >= MAX_LINE) len = MAX_LINE - 1;
                memcpy(current_section->items[current_section->count].text, line, len);
                current_section->items[current_section->count].text[len] = '\0';
                current_section->items[current_section->count].is_completed = 
                    is_completed_task(line);
                current_section->count++;
            }
        }
        else if (current_section != NULL) {
            // Non-task line in a section (preserve it)
            if (current_section->count < MAX_ITEMS && strlen(trim(line)) > 0) {
                size_t len = strlen(line);
                if (len >= MAX_LINE) len = MAX_LINE - 1;
                memcpy(current_section->items[current_section->count].text, line, len);
                current_section->items[current_section->count].text[len] = '\0';
                current_section->items[current_section->count].is_completed = 0;
                current_section->count++;
            }
        }
    }

    // Now write organized output
    
    // 1. Write preamble
    fputs(preamble, output);
    
    // 2. Write software TODO section (incomplete items only)
    fputs("\n", output);
    fputs(software_todo.header, output);
    for (int i = 0; i < software_todo.count; i++) {
        if (!software_todo.items[i].is_completed) {
            fputs(software_todo.items[i].text, output);
        }
    }
    
    // 3. Write middle content
    fputs(middle_content, output);
    
    // 4. Write analysis TODO section (incomplete items only)
    fputs("\n", output);
    fputs(analysis_todo.header, output);
    for (int i = 0; i < analysis_todo.count; i++) {
        if (!analysis_todo.items[i].is_completed) {
            fputs(analysis_todo.items[i].text, output);
        }
    }
    
    // 5. Write completed software goals
    fputs("\n", output);
    fputs(software_completed.header, output);
    // First add newly completed items from todo section (newest at top)
    for (int i = 0; i < software_todo.count; i++) {
        if (software_todo.items[i].is_completed) {
            fputs(software_todo.items[i].text, output);
        }
    }
    // Then add previously completed items (older ones at bottom)
    for (int i = 0; i < software_completed.count; i++) {
        fputs(software_completed.items[i].text, output);
    }
    
    // 6. Write completed analysis goals
    fputs("\n", output);
    fputs(analysis_completed.header, output);
    // First add newly completed items from todo section (newest at top)
    for (int i = 0; i < analysis_todo.count; i++) {
        if (analysis_todo.items[i].is_completed) {
            fputs(analysis_todo.items[i].text, output);
        }
    }
    // Then add previously completed items (older ones at bottom)
    for (int i = 0; i < analysis_completed.count; i++) {
        fputs(analysis_completed.items[i].text, output);
    }

    fclose(input);
    fclose(output);

    // Replace original file with organized version
    if (remove(input_file) != 0) {
        perror("Error removing original todo.md");
        return 1;
    }
    
    if (rename(temp_file, input_file) != 0) {
        perror("Error renaming temporary file");
        return 1;
    }

    // printf("Successfully reorganized todo.md\n");
    // printf("Completed tasks moved to bottom sections.\n");

    return 0;
}