# Task 1: Matrix Summation Performance Comparison

This task investigates the execution time of matrix summation using two different traversal methods:
- row-by-row traversal;
- column-by-column traversal.

The goal is to compare their performance for matrices of different sizes.

## Results from Console Output

```
Size	Sum1 time	Sum2 time	Sum
------------------------------------------
 500	0.000454	0.000423	250000
1000	0.001727	0.001667	1000000
1500	0.004218	0.005108	2250000
2000	0.007595	0.008964	4000000
2500	0.012227	0.012797	6250000
3000	0.014924	0.018625	9000000
4000	0.026766	0.033107	16000000
5000	0.041802	0.053294	25000000
```

## Conclusion from the Output

The execution time increases with matrix size for both algorithms.

For large matrices, the row-by-row traversal (`sum1`) works faster than the column-by-column traversal (`sum2`). This is because matrices in C are stored in memory by rows, so sequential row access uses CPU cache more efficiently.

# Task 2: Matrix Multiplication Performance Comparison

This task compares the execution time of six matrix multiplication algorithms for square matrices of different sizes.

The algorithms differ in the order of loops and memory access:
- `M1` and `M2` use the classic matrix multiplication approach with different row/column traversal order;
- `M3` and `M4` store one value from matrix `B` and reuse it inside the inner loop;
- `M5` and `M6` store one value from matrix `A` and process matrix rows more sequentially;

## Results from Console Output

```
Size	M1		M2		M3		M4		M5		M6
------------------------------------------------------------------------------------------------
  50	0.000170	0.000166	0.000207	0.000208	0.000192	0.000192
 100	0.001341	0.001406	0.001735	0.001632	0.001612	0.001583
 150	0.004769	0.004645	0.005391	0.005001	0.005346	0.005055
 200	0.011035	0.010974	0.012564	0.012336	0.012752	0.011825
 250	0.021451	0.021520	0.025389	0.024668	0.025614	0.022978
 300	0.037150	0.037822	0.042913	0.045652	0.044315	0.039740
 400	0.088996	0.090130	0.112798	0.106466	0.103379	0.094756
 500	0.174255	0.175889	0.228589	0.224012	0.204135	0.185442
```

## Conclusion from the Output

The execution time increases quickly as the matrix size grows, which matches the expected `O(n^3)` complexity.

For small matrices, the difference between algorithms is small, but for larger matrices it becomes more noticeable.
`M1`, `M2`, and `M6` show the best performance, while `M3` and `M4` are usually slower because of less efficient memory access.
