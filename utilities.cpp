//========================================================================
// Utilities
//========================================================================
// @brief: helper functions

#include "utilities.h"

// read file
int *read_map(char *filename)
{
    int n = 0;
    int *map = 0;
    char *str;
    FILE *file = fopen("filename", "r");
    if(!file)
    {   // open error
        file_error(filename);
    }
    while ((str=fgetl(file)))
    {
        n++;
        map = (int *)realloc(map, n*sizeof(int));
        map[n-1] = atoi(str);
    }
    return map;
}

// report open file error
void file_error(char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0); // exit(0): normal exit
}

// remove space, tab, and enter from string
void strip(char *s)
{
    size_t len = strlen(s);
    size_t offset = 0;

    for (size_t i = 0; i < len; i++)
    {
        char c = s[i];
        if (c == ' ' || c == '\t' || c == '\n')
        {
            offset++;
        }
        else
        {
            s[i-offset] = c;
        }
    }
    s[len-offset] = '\0';
}

// read one line from file
char *fgetl(FILE *file)
{
    if (feof(file))
    {   // check end of file indicator
        return 0;
    }
    size_t size = 512;
    char *line = (char *)malloc(size * sizeof(char));
    // read one line from file
    if (!fgets(line, size, file))
    {   // blank line
        free(line);
        return 0;
    }

    size_t curr = strlen(line);
    // verify the size of input line
    while (line[curr-1] != '\n' && !feof(file))
    {   // last char in line is '\0', size of line is 511
        if (curr == size - 1)
        {
            size *= 2;
            line = (char *)realloc(line, size * sizeof(char));
            if(!line)
            {
                printf("Required size:%ld\n", size);
                malloc_error();
            }
        }
        // read extra chars
        size_t readsize = size - curr;
        // check whether it overflow the maximum size
        // INT_MAX: 32767 (2^15-1) or greater*
        if (readsize > INT_MAX)
        {
            readsize = INT_MAX - 1;
        }
        // continue to read this line
        fgets(&line[curr], readsize, file);
        // update the current length read from file
        curr = strlen(line);
    }
    if (line[curr - 1] == '\n')
    {   // final line in char - '\0'
        line[curr - 1] = '\0';
    }

    return line;
}

// report malloc error
void malloc_error()
{
    fprintf(stderr, "Malloc error\n");
    exit(-1); // exit(other numbers): abnormal exit
}

// report specific error
void error(const char *s)
{
    printf("%s\n",s);
    assert(0);
    exit(-1);
}

// free array of pointer
void free_ptrs(void **ptrs, int n)
{
    for (int i = 0; i < n; i++)
    {
        free(ptrs[i]);
    }
    free(ptrs);
}

// find the maximum value in an array, return its index
int max_index(float *a, int n)
{
    if(n <= 0)
    {
        return -1;
    }
    int max_index = 0;
    float temp = a[0];
    // traverse the array
    for (int i = 1; i < n; i++)
    {
        if (a[i] > temp)
        {
            temp = a[i];
            max_index = i;
        }
    }
    return max_index;
}

//return a random number in the given range(min, max)
float rand_uniform(float min, float max)
{
    if(max < min)
    {
        float temp = min;
        min = max;
        max = temp;
    }
    return ((float)rand()/RAND_MAX * (max-min)) + min;
}

// print function
void print_statistics(float *a, int n)
{
    float m = mean_array(a, n);
    float v = variance_array(a, n);
    printf("MSE: %.6f, Mean: %.6f, Variance: %.6f\n", mse_array(a, n), m, v);
}

// mean value of array a
float mean_array(float *a, int n)
{
    return sum_array(a, n)/n;
}

// sum of array a
float sum_array(float *a, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += a[i];
    }
    return sum;
}

// variance of array a
float variance_array(float *a, int n)
{
    float sum = 0;
    float mean = mean_array(a, n);
    for (int i = 0; i < n; i++)
    {
        sum += (a[i] - mean)*(a[i] - mean);
    }
    return sum/n;
}

// mean squared error of array a
float mse_array(float *a, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += a[i] * a[i];
    }
    return sqrt(sum/n);
}

// difference of two squares
float mag_array(float *a, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}
