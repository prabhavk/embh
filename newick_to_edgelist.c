/* 

converts newick to edge list
input to embh is file in edge list format

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_NAME 256
#define MAX_EDGES 10000

typedef struct {
    char node1[MAX_NAME];
    char node2[MAX_NAME];
    double length;
} Edge;

typedef struct {
    Edge edges[MAX_EDGES];
    int edge_count;
    int hidden_count;
} Graph;

void trim(char* str) {
    char* end;
    while (isspace(*str)) str++;
    if (*str == 0) return;
    end = str + strlen(str) - 1;
    while (end > str && isspace(*end)) end--;
    *(end + 1) = 0;
}

void add_edge(Graph* g, const char* node1, const char* node2, double length) {
    if (g->edge_count >= MAX_EDGES) {
        fprintf(stderr, "Error: too many edges\n");
        exit(1);
    }
    strcpy(g->edges[g->edge_count].node1, node1);
    strcpy(g->edges[g->edge_count].node2, node2);
    g->edges[g->edge_count].length = length;
    g->edge_count++;
}

char* parse_newick(char* str, Graph* g, char* parent) {
    char name[MAX_NAME] = "";
    char current_node[MAX_NAME];
    double length = 0.0;
    int is_leaf = 1;
    
    while (*str && isspace(*str)) str++;
    
    if (*str == '(') {
        is_leaf = 0;
        str++;
        sprintf(current_node, "h_%d", g->hidden_count++);
        
        while (*str != ')') {
            str = parse_newick(str, g, current_node);
            while (*str && isspace(*str)) str++;
            if (*str == ',') {
                str++;
                while (*str && isspace(*str)) str++;
            }
        }
        str++;
    }
    
    while (*str && isspace(*str)) str++;
    
    if (*str && *str != ':' && *str != ',' && *str != ')' && *str != ';') {
        int i = 0;
        while (*str && *str != ':' && *str != ',' && *str != ')' && *str != ';' && i < MAX_NAME - 1) {
            name[i++] = *str++;
        }
        name[i] = '\0';
        trim(name);
        if (is_leaf) {
            strcpy(current_node, name);
        }
    }
    
    while (*str && isspace(*str)) str++;
    
    if (*str == ':') {
        str++;
        while (*str && isspace(*str)) str++;
        length = strtod(str, &str);
    }
    
    if (parent && parent[0] != '\0') {
        add_edge(g, parent, current_node, length);
    }
    
    return str;
}

void read_newick_file(const char* filename, Graph* g) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        exit(1);
    }
    
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    char* buffer = (char*)malloc(size + 1);
    if (!buffer) {
        fprintf(stderr, "Error: memory allocation failed\n");
        exit(1);
    }
    
    fread(buffer, 1, size, fp);
    buffer[size] = '\0';
    fclose(fp);
    
    g->edge_count = 0;
    g->hidden_count = 0;
    
    char* ptr = buffer;
    while (*ptr && isspace(*ptr)) ptr++;
    
    parse_newick(ptr, g, NULL);
    
    free(buffer);
}

void write_edge_list(const char* filename, Graph* g) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: cannot open output file %s\n", filename);
        exit(1);
    }
    
    for (int i = 0; i < g->edge_count; i++) {
        fprintf(fp, "%s\t%s\t%.6f\n", 
                g->edges[i].node1, 
                g->edges[i].node2, 
                g->edges[i].length);
    }
    
    fclose(fp);
}

int main(int argc, char* argv[]) {
    char* input_file = NULL;
    char* output_file = NULL;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        }
    }
    
    if (!input_file || !output_file) {
        fprintf(stderr, "Usage: %s -i input.newick -o output.txt\n", argv[0]);
        return 1;
    }
    
    Graph g;
    read_newick_file(input_file, &g);
    write_edge_list(output_file, &g);
    
    printf("Converted %d edges from %s to %s\n", g.edge_count, input_file, output_file);
    
    return 0;
}