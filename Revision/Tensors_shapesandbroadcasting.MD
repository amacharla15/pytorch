# PyTorch Fundamentals Review — Day 1: Tensors, Shapes, Broadcasting, and Dataset Representation

This repository documents a structured PyTorch fundamentals review focused on tensor reasoning, dataset representation, broadcasting, reshaping, dtype choices, indexing, and matrix multiplication. The goal is to build strong interview-level fluency for reading, debugging, and explaining PyTorch tensor operations.

## Focus Areas

- Tensor shapes and dimensions
- Dataset-to-tensor representation
- Rows as examples and columns as features
- Broadcasting rules
- Matrix multiplication vs elementwise operations
- Regression vs classification tensor shapes
- dtype choices for inputs, labels, logits, and targets
- Reshape/view intuition
- Indexing and slicing
- Interview-style reasoning for common tensor errors

---

## Toy Dataset: House Price / House Category Prediction

The following table represents a small tabular dataset.

Each row is one house.  
Each column is one feature.  
The target can either be a continuous price for regression or a class label for classification.

| house_id | sqft | bedrooms | bathrooms | age_years | distance_miles | school_rating | lot_sqft | garage_spaces | price_k |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1200 | 2 | 1 | 30 | 8.5 | 6 | 3500 | 1 | 310 |
| 2 | 1800 | 3 | 2 | 15 | 5.2 | 8 | 5000 | 2 | 475 |
| 3 | 950 | 2 | 1 | 45 | 12.0 | 5 | 2800 | 0 | 250 |
| 4 | 2200 | 4 | 3 | 10 | 3.1 | 9 | 6200 | 2 | 620 |
| 5 | 1600 | 3 | 2 | 20 | 6.8 | 7 | 4500 | 1 | 410 |
| 6 | 3000 | 5 | 4 | 5 | 2.0 | 10 | 8000 | 3 | 850 |
| 7 | 1400 | 3 | 1 | 25 | 9.3 | 6 | 3900 | 1 | 360 |
| 8 | 2600 | 4 | 3 | 8 | 4.0 | 9 | 7000 | 2 | 710 |
| 9 | 1100 | 2 | 1 | 35 | 10.5 | 5 | 3200 | 1 | 295 |
| 10 | 2000 | 3 | 2 | 12 | 5.5 | 8 | 5400 | 2 | 530 |

---

## Dataset Tensor Representation

The input feature matrix is represented as:

text
X shape = [10, 8]

Meaning:

10 examples / houses
8 input features per house

The 8 input features are:

sqft
bedrooms
bathrooms
age_years
distance_miles
school_rating
lot_sqft
garage_spaces

The regression target is:

y_reg shape = [10] or [10, 1]

Meaning:

one price target per house

For classification, if houses are categorized as:

0 = cheap
1 = medium
2 = expensive

Then classification labels are:

labels_cls shape = [10]

Meaning:

one class ID per house
Core Shape Meaning
Tensor	Shape	Meaning
X	[10, 8]	10 examples, 8 features per example
One row of X	[8]	One house with all 8 feature values
One column of X	[10]	One feature across all 10 houses
feature_mean	[8]	One mean value per feature column
row_bonus	[10, 1]	One value per row/example
y_reg	[10] or [10, 1]	One regression target per example
labels_cls	[10]	One class index per example
W_reg	[8, 1]	Maps 8 features to 1 regression output
W_cls	[8, 3]	Maps 8 features to 3 class scores
logits	[10, 3]	3 raw class scores per example
Broadcasting Rules

Broadcasting allows PyTorch to apply operations between tensors of different shapes when their dimensions are compatible.

Rule:

Compare shapes from the right.
Dimensions are compatible if:
1. they are equal, or
2. one of them is 1.
Feature-wise Broadcasting
X shape            = [10, 8]
feature_mean shape = [8]

This is valid.

PyTorch treats [8] like [1, 8], then broadcasts it across the 10 rows.

X - feature_mean -> [10, 8]

Meaning:

subtract one feature mean from each feature column for every example
Row-wise Broadcasting
X shape         = [10, 8]
row_bonus shape = [10, 1]

This is valid.

The 1 in the second dimension broadcasts across the 8 features.

X + row_bonus -> [10, 8]

Meaning:

apply one row-specific value across all features of each row
Invalid Broadcasting Example
X shape      = [10, 8]
labels shape = [10]

This is not valid for direct addition.

Alignment from the right:

X:      [10, 8]
labels:     [10]

The last dimensions are:

8 vs 10

They are not equal, and neither is 1.

So:

X + labels -> invalid

Even though labels logically contain one value per example, shape [10] does not broadcast with [10, 8] as row-wise values. For row-wise broadcasting, the shape should be [10, 1].

Matrix Multiplication vs Elementwise Operations

Elementwise operations such as +, -, and * use broadcasting rules.

Matrix multiplication uses inner-dimension matching:

[m, n] @ [n, p] = [m, p]
Regression Example
X shape     = [10, 8]
W_reg shape = [8, 1]

Matrix multiplication:

X @ W_reg = [10, 1]

Meaning:

10 houses
1 predicted price per house
Classification Example
X shape     = [10, 8]
W_cls shape = [8, 3]

Matrix multiplication:

X @ W_cls = [10, 3]

Meaning:

10 houses
3 class scores per house

Each row of [10, 3] contains raw scores for:

cheap
medium
expensive
Dtype Review

Different tensors have different mathematical roles, so their dtypes matter.

Task	Tensor	Shape	Usual dtype	Reason
Regression	X	[10, 8]	float32	Input features are continuous numeric values
Regression	y_reg	[10] or [10, 1]	float32	Price is a continuous target
Classification	X	[10, 8]	float32	Model inputs need floating-point math
Classification	logits	[10, 3]	float32	Logits are continuous model scores
Classification	labels_cls	[10]	int64 / long	Labels are class indices
Masking	mask	depends	bool	Masks represent true/false positions

Important distinction:

logits / predictions = model outputs
labels / targets = correct answers

For classification, labels are usually class IDs:

0 = cheap
1 = medium
2 = expensive

So labels should be integer class indices, not floating-point prediction scores.

Reshape / View Intuition

A reshape is mechanically valid if the total number of values stays the same.

For:

X shape = [10, 8]

Total values:

10 * 8 = 80

Valid reshapes:

[10, 8] -> [80]
[10, 8] -> [5, 16]
[10, 8] -> [8, 10]
[10, 8] -> [10, 1, 8]

Invalid reshape:

[10, 8] -> [10, 10]

because:

10 * 8 = 80
10 * 10 = 100
Mechanical Validity vs Semantic Meaning

A reshape can be valid by value count but still wrong for the dataset meaning.

Example:

[10, 8] -> [5, 16]

This is mechanically valid because both shapes contain 80 values.

However, for a tabular dataset where [10, 8] means 10 houses and 8 features per house, reshaping to [5, 16] usually destroys the original example-feature structure.

A safer reshape is:

[10, 8] -> [10, 1, 8]

This keeps the 10 examples and 8 features together while inserting a size-1 dimension.

Reshape vs Transpose
[10, 8] -> [8, 10]

as a reshape is not the same as a transpose.

Reshape:

flattens and regroups values

Transpose:

swaps dimensions while preserving row-column relationships
Indexing and Slicing Review

For:

X shape = [10, 8]
Expression	Meaning	Shape
X[0]	First house, all 8 features	[8]
X[:, 0]	Feature column 0 for all houses	[10]
X[:, 2]	Column index 2, third feature, for all houses	[10]
X[0, 0]	One scalar value: row 0, column 0	scalar
X[0:5]	First 5 houses	[5, 8]
X[:, 0:2]	All houses, columns 0 and 1	[10, 2]
X[:, 0:3]	All houses, columns 0, 1, and 2	[10, 3]
X[2:6, 1:4]	Rows 2-5 and columns 1-3	[4, 3]

Important indexing rule:

Python slicing excludes the ending index.

So:

0:3 selects 0, 1, 2
2:6 selects 2, 3, 4, 5

Also:

X[:, 2]

means column index 2, which is the third column, not the second column.

Interview-Style Questions and Answers
1. What does X shape = [10, 8] mean?

It means there are 10 examples and 8 features per example. In table form, each row is one house and each column is one feature.

2. Why is X - feature_mean valid when feature_mean shape = [8]?

Because [8] represents one value per feature column. PyTorch treats it like [1, 8] and broadcasts it across the 10 examples.

Output shape:

[10, 8]
3. Why is X + labels_cls invalid when labels_cls shape = [10]?

Broadcasting aligns from the right:

X:          [10, 8]
labels_cls:    [10]

The last dimensions are 8 and 10, which are incompatible.

So direct addition is invalid.

4. Why is X @ W_reg valid when W_reg shape = [8, 1]?

Matrix multiplication checks inner dimensions.

[10, 8] @ [8, 1] = [10, 1]

The inner dimensions are both 8, so the operation is valid.

The output means one regression prediction per house.

5. Why is X @ W_cls valid when W_cls shape = [8, 3]?
[10, 8] @ [8, 3] = [10, 3]

The result contains 3 class scores per example.

For 10 houses and 3 categories, the output shape is:

[10, 3]
6. Why should classification labels be int64 / long?

Classification labels represent class indices, such as:

0 = cheap
1 = medium
2 = expensive

They are not continuous values. Standard classification losses expect class IDs as integer labels.

7. Why should logits be float32?

Logits are continuous model scores. They are produced by floating-point model weights and are used in gradient-based training. Floating-point values are required for meaningful loss and gradient computation.

8. Can [10, 8] be reshaped to [5, 16]?

Mechanically, yes.

10 * 8 = 5 * 16 = 80

But for a tabular dataset, this is usually not meaningful because it mixes the original row/example and column/feature structure.

9. Can [10, 8] be reshaped to [10, 1, 8]?

Yes.

10 * 8 = 10 * 1 * 8 = 80

This is often safer than [5, 16] because it preserves the 10 examples and 8 features while inserting a size-1 dimension.

10. What is the difference between reshape and transpose?

Reshape changes how values are grouped after flattening.

Transpose swaps axes while preserving the relationship between rows and columns.

They can produce the same final shape but different value arrangements.