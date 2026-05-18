#include <stdio.h>
#include <time.h>

#define MAX_SIZE 5000

long long sum1(int A[][MAX_SIZE], const int m, const int n)
{
    long long S = 0;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            S += A[i][j];
        }
    }

    return S;
}

long long sum2(int A[][MAX_SIZE], const int m, const int n)
{
    long long S = 0;

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            S += A[i][j];
        }
    }

    return S;
}

double measure_time(long long (*func)(int[][MAX_SIZE], int, int),
                    int A[][MAX_SIZE], const int m, const int n, long long *result)
{
    const clock_t start = clock();
    *result = func(A, m, n);
    const clock_t end = clock();

    return (double)(end - start) / CLOCKS_PER_SEC;
}

int main(void)
{
    static int A[MAX_SIZE][MAX_SIZE];

    const int sizes[] = {500, 1000, 1500, 2000, 2500, 3000, 4000, 5000};
    constexpr int count = sizeof(sizes) / sizeof(sizes[0]);

    printf("Size\tSum1 time\tSum2 time\tSum\n");
    printf("------------------------------------------\n");

    for (int t = 0; t < count; t++)
    {
        const int n = sizes[t];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i][j] = 1;
            }
        }

        long long result1, result2;

        const double time1 = measure_time(sum1, A, n, n, &result1);
        const double time2 = measure_time(sum2, A, n, n, &result2);

        printf("%4d\t%.6f\t%.6f\t%lld\n",
               n, time1, time2, result1);
    }

    return 0;
}