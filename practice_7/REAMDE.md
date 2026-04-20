# Task 1

## Polynomial Regression: Degree Comparison

![Original Data](results/task_1_original.png)
![Polynomial Degree 2](results/task_1_poly_degree_2.png)
![Polynomial Degree 3](results/task_1_poly_degree_3.png)
![Polynomial Degree 4](results/task_1_poly_degree_4.png)
![Polynomial Degree 5](results/task_1_poly_degree_5.png)

## Results from Console Output
```
Degree: 2
	Accuracy:  0.9810
	Precision: 1.0000
	Recall:    0.9615
	F1-Score:  0.9804

Degree: 3
	Accuracy:  0.5210
	Precision: 1.0000
	Recall:    0.0284
	F1-Score:  0.0552

Degree: 4
	Accuracy:  0.9230
	Precision: 1.0000
	Recall:    0.8438
	F1-Score:  0.9153

Degree: 5
	Accuracy:  0.5130
	Precision: 1.0000
	Recall:    0.0122
	F1-Score:  0.0240
```

# Task 2

## SVM Kernels with Different Noise Levels

### Noise Level 0.0
![Original (Noise 0.0)](results/task_2_original_noise_0.0.png)
![Linear (Noise 0.0)](results/task_2_linear_noise_0.0.png)
![RBF (Noise 0.0)](results/task_2_rbf_noise_0.0.png)
![Polynomial (Noise 0.0)](results/task_2_polynomial_noise_0.0.png)
![Sigmoid (Noise 0.0)](results/task_2_sigmoid_noise_0.0.png)

### Noise Level 0.1
![Original (Noise 0.1)](results/task_2_original_noise_0.1.png)
![Linear (Noise 0.1)](results/task_2_linear_noise_0.1.png)
![RBF (Noise 0.1)](results/task_2_rbf_noise_0.1.png)
![Polynomial (Noise 0.1)](results/task_2_polynomial_noise_0.1.png)
![Sigmoid (Noise 0.1)](results/task_2_sigmoid_noise_0.1.png)

### Noise Level 0.2
![Original (Noise 0.2)](results/task_2_original_noise_0.2.png)
![Linear (Noise 0.2)](results/task_2_linear_noise_0.2.png)
![RBF (Noise 0.2)](results/task_2_rbf_noise_0.2.png)
![Polynomial (Noise 0.2)](results/task_2_polynomial_noise_0.2.png)
![Sigmoid (Noise 0.2)](results/task_2_sigmoid_noise_0.2.png)

### Noise Level 0.3
![Original (Noise 0.3)](results/task_2_original_noise_0.3.png)
![Linear (Noise 0.3)](results/task_2_linear_noise_0.3.png)
![RBF (Noise 0.3)](results/task_2_rbf_noise_0.3.png)
![Polynomial (Noise 0.3)](results/task_2_polynomial_noise_0.3.png)
![Sigmoid (Noise 0.3)](results/task_2_sigmoid_noise_0.3.png)

## Results from Console Output
```
Noise Level: 0.0:
	Linear:
		accuracy: 0.425
		precision: 0.4251293297345929
		recall: 0.43999999999999995
		f1: 0.4287701873514231

	RBF:
		accuracy: 1.0
		precision: 1.0
		recall: 1.0
		f1: 1.0

	Polynomial:
		accuracy: 0.495
		precision: 0.491827235944883
		recall: 0.69
		f1: 0.5716171338511764

	Sigmoid:
		accuracy: 0.45999999999999996
		precision: 0.44853059581320454
		recall: 0.47000000000000003
		f1: 0.4543396674103203

Noise Level: 0.1:
	Linear:
		accuracy: 0.42000000000000004
		precision: 0.4203512585812357
		recall: 0.43
		f1: 0.42242496521566286

	RBF:
		accuracy: 0.79
		precision: 0.7955411255411257
		recall: 0.78
		f1: 0.7844250871080141

	Polynomial:
		accuracy: 0.505
		precision: 0.5034903567161632
		recall: 0.71
		f1: 0.588788068418857

	Sigmoid:
		accuracy: 0.475
		precision: 0.47733929685874765
		recall: 0.49000000000000005
		f1: 0.4828767356674334

Noise Level: 0.2:
	Linear:
		accuracy: 0.4
		precision: 0.3993719806763285
		recall: 0.4
		f1: 0.3968856602393934

	RBF:
		accuracy: 0.6599999999999999
		precision: 0.6481966739681102
		recall: 0.6900000000000001
		f1: 0.6654207696068161

	Polynomial:
		accuracy: 0.535
		precision: 0.5314452214452214
		recall: 0.6799999999999999
		f1: 0.5921418805422086

	Sigmoid:
		accuracy: 0.48999999999999994
		precision: 0.48550138026224987
		recall: 0.55
		f1: 0.5119905759176345

Noise Level: 0.3:
	Linear:
		accuracy: 0.38499999999999995
		precision: 0.3846723528188059
		recall: 0.38
		f1: 0.3802777067893347

	RBF:
		accuracy: 0.5900000000000001
		precision: 0.5720756219003063
		recall: 0.72
		f1: 0.6351395011538948

	Polynomial:
		accuracy: 0.5050000000000001
		precision: 0.5073631047315258
		recall: 0.6
		f1: 0.5413735387419598

	Sigmoid:
		accuracy: 0.48
		precision: 0.4727142857142857
		recall: 0.5
		f1: 0.4836778939217964
```

# Task 3

## SVM Kernel Comparison on Moons Dataset

![Original Data](results/task_3_original.png)
![Linear Kernel](results/task_3_linear.png)
![RBF Kernel](results/task_3_rbf.png)
![Polynomial Kernel](results/task_3_polynomial.png)
![Sigmoid Kernel](results/task_3_sigmoid.png)

## Results from Console Output
```
Linear:
	accuracy: 0.8699999999999999
	precision: 0.877439827189457
	recall: 0.8700000000000001
	f1: 0.8715815510086639

RBF:
	accuracy: 1.0
	precision: 1.0
	recall: 1.0
	f1: 1.0

Polynomial:
	accuracy: 0.93
	precision: 0.890053569618787
	recall: 0.99
	f1: 0.9355576968018704

Sigmoid:
	accuracy: 0.635
	precision: 0.6366236103078208
	recall: 0.65
	f1: 0.6348267876064349
```