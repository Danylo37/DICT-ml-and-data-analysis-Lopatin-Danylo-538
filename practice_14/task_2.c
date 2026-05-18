#include <stdio.h>
#include <time.h>

#define MAX_SIZE 500

void multiply1(int A[][MAX_SIZE], int B[][MAX_SIZE], int C[][MAX_SIZE], const int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int sum = 0;
            for (int k = 0; k < n; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] += sum;
        }
    }
}

void multiply2(int A[][MAX_SIZE], int B[][MAX_SIZE], int C[][MAX_SIZE], const int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            int sum = 0;
            for (int k = 0; k < n; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] += sum;
        }
    }
}

void multiply3(int A[][MAX_SIZE], int B[][MAX_SIZE], int C[][MAX_SIZE], const int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int k = 0; k < n; k++)
        {
            const int r = B[k][j];
            for (int i = 0; i < n; i++)
            {
                C[i][j] += A[i][k] * r;
            }
        }
    }
}

void multiply4(int A[][MAX_SIZE], int B[][MAX_SIZE], int C[][MAX_SIZE], const int n)
{
    for (int k = 0; k < n; k++)
    {
        for (int j = 0; j < n; j++)
        {
            const int r = B[k][j];
            for (int i = 0; i < n; i++)
            {
                C[i][j] += A[i][k] * r;
            }
        }
    }
}

void multiply5(int A[][MAX_SIZE], int B[][MAX_SIZE], int C[][MAX_SIZE], const int n)
{
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < n; i++)
        {
            const int r = A[i][k];
            for (int j = 0; j < n; j++)
            {
                C[i][j] += r * B[k][j];
            }
        }
    }
}

void multiply6(int A[][MAX_SIZE], int B[][MAX_SIZE], int C[][MAX_SIZE], const int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            const int r = A[i][k];
            for (int j = 0; j < n; j++)
            {
                C[i][j] += r * B[k][j];
            }
        }
    }
}

double measure_time(void (*func)(int[][MAX_SIZE], int[][MAX_SIZE], int[][MAX_SIZE], int),
                    int A[][MAX_SIZE], int B[][MAX_SIZE], int C[][MAX_SIZE], const int n)
{
    const clock_t start = clock();
    func(A, B, C, n);
    const clock_t end = clock();

    return (double)(end - start) / CLOCKS_PER_SEC;
}

void fill_matrix(int M[][MAX_SIZE], const int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            M[i][j] = 1;
        }
    }
}

void clear_matrix(int M[][MAX_SIZE], const int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            M[i][j] = 0;
        }
    }
}

int main(void)
{
    static int A[MAX_SIZE][MAX_SIZE];
    static int B[MAX_SIZE][MAX_SIZE];
    static int C[MAX_SIZE][MAX_SIZE];

    const int sizes[] = {50, 100, 150, 200, 250, 300, 400, 500};
    constexpr int count = sizeof(sizes) / sizeof(sizes[0]);

    printf("Size\tM1\t\tM2\t\tM3\t\tM4\t\tM5\t\tM6\n");
    printf("------------------------------------------------------------------------------------------------\n");

    for (int t = 0; t < count; t++)
    {
        const int n = sizes[t];

        fill_matrix(A, n);
        fill_matrix(B, n);

        clear_matrix(C, n);
        const double time1 = measure_time(multiply1, A, B, C, n);

        clear_matrix(C, n);
        const double time2 = measure_time(multiply2, A, B, C, n);

        clear_matrix(C, n);
        const double time3 = measure_time(multiply3, A, B, C, n);

        clear_matrix(C, n);
        const double time4 = measure_time(multiply4, A, B, C, n);

        clear_matrix(C, n);
        const double time5 = measure_time(multiply5, A, B, C, n);

        clear_matrix(C, n);
        const double time6 = measure_time(multiply6, A, B, C, n);

        printf("%4d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
               n, time1, time2, time3, time4, time5, time6);
    }

    return 0;
}
