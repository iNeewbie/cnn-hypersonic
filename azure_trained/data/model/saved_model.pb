њм
шќ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Щ
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.6.22v2.6.1-9-gc2363d6d0258÷≠

conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ц*
shared_nameconv2d/kernel
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*'
_output_shapes
:Ц*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ц*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:Ц*
dtype0
Д
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ЦЦ* 
shared_nameconv2d_1/kernel
}
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*(
_output_shapes
:ЦЦ*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ц*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:Ц*
dtype0
Д
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ЦЦ* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:ЦЦ*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ц*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:Ц*
dtype0
Д
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ЦЦ* 
shared_nameconv2d_3/kernel
}
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:ЦЦ*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ц*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:Ц*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЏЎ*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
ЏЎ*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ў*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:Ў*
dtype0
Д
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ЦЦ* 
shared_nameconv2d_4/kernel
}
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*(
_output_shapes
:ЦЦ*
dtype0
s
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ц*
shared_nameconv2d_4/bias
l
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:Ц*
dtype0
Д
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ЦЦ* 
shared_nameconv2d_5/kernel
}
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*(
_output_shapes
:ЦЦ*
dtype0
s
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ц*
shared_nameconv2d_5/bias
l
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes	
:Ц*
dtype0
Д
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ЦЦ* 
shared_nameconv2d_6/kernel
}
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*(
_output_shapes
:ЦЦ*
dtype0
s
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ц*
shared_nameconv2d_6/bias
l
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes	
:Ц*
dtype0
Д
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ЦЦ* 
shared_nameconv2d_7/kernel
}
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*(
_output_shapes
:ЦЦ*
dtype0
s
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ц*
shared_nameconv2d_7/bias
l
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes	
:Ц*
dtype0
Г
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ц* 
shared_nameconv2d_8/kernel
|
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*'
_output_shapes
:Ц*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:*
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0

NoOpNoOp
аc
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ыc
valueСcBОc BЗc
Б
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer_with_weights-6
layer-17
layer-18
layer_with_weights-7
layer-19
layer-20
layer_with_weights-8
layer-21
layer-22
layer-23
layer_with_weights-9
layer-24
layer-25
layer-26
layer-27
layer-28
	optimizer
regularization_losses
 trainable_variables
!	variables
"	keras_api
#
signatures
 
h

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
R
*regularization_losses
+trainable_variables
,	variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
R
4regularization_losses
5trainable_variables
6	variables
7	keras_api
h

8kernel
9bias
:regularization_losses
;trainable_variables
<	variables
=	keras_api
R
>regularization_losses
?trainable_variables
@	variables
A	keras_api
h

Bkernel
Cbias
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
 
R
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
R
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
R
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
R
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
h

Xkernel
Ybias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
R
^regularization_losses
_trainable_variables
`	variables
a	keras_api
h

bkernel
cbias
dregularization_losses
etrainable_variables
f	variables
g	keras_api
R
hregularization_losses
itrainable_variables
j	variables
k	keras_api
h

lkernel
mbias
nregularization_losses
otrainable_variables
p	variables
q	keras_api
R
rregularization_losses
strainable_variables
t	variables
u	keras_api
h

vkernel
wbias
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
R
|regularization_losses
}trainable_variables
~	variables
	keras_api
n
Аkernel
	Бbias
Вregularization_losses
Гtrainable_variables
Д	variables
Е	keras_api
V
Жregularization_losses
Зtrainable_variables
И	variables
Й	keras_api

К	keras_api
n
Лkernel
	Мbias
Нregularization_losses
Оtrainable_variables
П	variables
Р	keras_api

С	keras_api
V
Тregularization_losses
Уtrainable_variables
Ф	variables
Х	keras_api
V
Цregularization_losses
Чtrainable_variables
Ш	variables
Щ	keras_api
V
Ъregularization_losses
Ыtrainable_variables
Ь	variables
Э	keras_api
:

Юdecay
Яlearning_rate
†momentum
	°iter
 
Ъ
$0
%1
.2
/3
84
95
B6
C7
X8
Y9
b10
c11
l12
m13
v14
w15
А16
Б17
Л18
М19
Ъ
$0
%1
.2
/3
84
95
B6
C7
X8
Y9
b10
c11
l12
m13
v14
w15
А16
Б17
Л18
М19
≤
Ґmetrics
regularization_losses
 trainable_variables
!	variables
£non_trainable_variables
 §layer_regularization_losses
•layer_metrics
¶layers
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
≤
Іmetrics
&regularization_losses
'trainable_variables
(	variables
®non_trainable_variables
 ©layer_regularization_losses
™layer_metrics
Ђlayers
 
 
 
≤
ђmetrics
*regularization_losses
+trainable_variables
,	variables
≠non_trainable_variables
 Ѓlayer_regularization_losses
ѓlayer_metrics
∞layers
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
≤
±metrics
0regularization_losses
1trainable_variables
2	variables
≤non_trainable_variables
 ≥layer_regularization_losses
іlayer_metrics
µlayers
 
 
 
≤
ґmetrics
4regularization_losses
5trainable_variables
6	variables
Јnon_trainable_variables
 Єlayer_regularization_losses
єlayer_metrics
Їlayers
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

80
91

80
91
≤
їmetrics
:regularization_losses
;trainable_variables
<	variables
Љnon_trainable_variables
 љlayer_regularization_losses
Њlayer_metrics
њlayers
 
 
 
≤
јmetrics
>regularization_losses
?trainable_variables
@	variables
Ѕnon_trainable_variables
 ¬layer_regularization_losses
√layer_metrics
ƒlayers
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1

B0
C1
≤
≈metrics
Dregularization_losses
Etrainable_variables
F	variables
∆non_trainable_variables
 «layer_regularization_losses
»layer_metrics
…layers
 
 
 
≤
 metrics
Hregularization_losses
Itrainable_variables
J	variables
Ћnon_trainable_variables
 ћlayer_regularization_losses
Ќlayer_metrics
ќlayers
 
 
 
≤
ѕmetrics
Lregularization_losses
Mtrainable_variables
N	variables
–non_trainable_variables
 —layer_regularization_losses
“layer_metrics
”layers
 
 
 
≤
‘metrics
Pregularization_losses
Qtrainable_variables
R	variables
’non_trainable_variables
 ÷layer_regularization_losses
„layer_metrics
Ўlayers
 
 
 
≤
ўmetrics
Tregularization_losses
Utrainable_variables
V	variables
Џnon_trainable_variables
 џlayer_regularization_losses
№layer_metrics
Ёlayers
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

X0
Y1
≤
ёmetrics
Zregularization_losses
[trainable_variables
\	variables
яnon_trainable_variables
 аlayer_regularization_losses
бlayer_metrics
вlayers
 
 
 
≤
гmetrics
^regularization_losses
_trainable_variables
`	variables
дnon_trainable_variables
 еlayer_regularization_losses
жlayer_metrics
зlayers
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

b0
c1

b0
c1
≤
иmetrics
dregularization_losses
etrainable_variables
f	variables
йnon_trainable_variables
 кlayer_regularization_losses
лlayer_metrics
мlayers
 
 
 
≤
нmetrics
hregularization_losses
itrainable_variables
j	variables
оnon_trainable_variables
 пlayer_regularization_losses
рlayer_metrics
сlayers
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1

l0
m1
≤
тmetrics
nregularization_losses
otrainable_variables
p	variables
уnon_trainable_variables
 фlayer_regularization_losses
хlayer_metrics
цlayers
 
 
 
≤
чmetrics
rregularization_losses
strainable_variables
t	variables
шnon_trainable_variables
 щlayer_regularization_losses
ъlayer_metrics
ыlayers
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

v0
w1

v0
w1
≤
ьmetrics
xregularization_losses
ytrainable_variables
z	variables
эnon_trainable_variables
 юlayer_regularization_losses
€layer_metrics
Аlayers
 
 
 
≤
Бmetrics
|regularization_losses
}trainable_variables
~	variables
Вnon_trainable_variables
 Гlayer_regularization_losses
Дlayer_metrics
Еlayers
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

А0
Б1

А0
Б1
µ
Жmetrics
Вregularization_losses
Гtrainable_variables
Д	variables
Зnon_trainable_variables
 Иlayer_regularization_losses
Йlayer_metrics
Кlayers
 
 
 
µ
Лmetrics
Жregularization_losses
Зtrainable_variables
И	variables
Мnon_trainable_variables
 Нlayer_regularization_losses
Оlayer_metrics
Пlayers
 
[Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Л0
М1

Л0
М1
µ
Рmetrics
Нregularization_losses
Оtrainable_variables
П	variables
Сnon_trainable_variables
 Тlayer_regularization_losses
Уlayer_metrics
Фlayers
 
 
 
 
µ
Хmetrics
Тregularization_losses
Уtrainable_variables
Ф	variables
Цnon_trainable_variables
 Чlayer_regularization_losses
Шlayer_metrics
Щlayers
 
 
 
µ
Ъmetrics
Цregularization_losses
Чtrainable_variables
Ш	variables
Ыnon_trainable_variables
 Ьlayer_regularization_losses
Эlayer_metrics
Юlayers
 
 
 
µ
Яmetrics
Ъregularization_losses
Ыtrainable_variables
Ь	variables
†non_trainable_variables
 °layer_regularization_losses
Ґlayer_metrics
£layers
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
(
§0
•1
¶2
І3
®4
 
 
 
ё
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

©total

™count
Ђ	variables
ђ	keras_api
8

≠total

Ѓcount
ѓ	variables
∞	keras_api
8

±total

≤count
≥	variables
і	keras_api
I

µtotal

ґcount
Ј
_fn_kwargs
Є	variables
є	keras_api
I

Їtotal

їcount
Љ
_fn_kwargs
љ	variables
Њ	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

©0
™1

Ђ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

≠0
Ѓ1

ѓ	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

±0
≤1

≥	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

µ0
ґ1

Є	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ї0
ї1

љ	variables
О
serving_default_input_1Placeholder*1
_output_shapes
:€€€€€€€€€ђђ*
dtype0*&
shape:€€€€€€€€€ђђ
z
serving_default_input_2Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
“
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/bias*!
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::€€€€€€€€€ђђ:€€€€€€€€€ђђ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_221076
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ы
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOpConst*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__traced_save_222212
о
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasdecaylearning_ratemomentumSGD/itertotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__traced_restore_222324®х
°
°
)__inference_conv2d_7_layer_call_fn_221967

inputs#
unknown:ЦЦ
	unknown_0:	Ц
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2203322
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€<<Ц: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€<<Ц
 
_user_specified_nameinputs
»
^
B__inference_lambda_layer_call_and_return_conditional_losses_220397

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2€
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
strided_slicet
IdentityIdentitystrided_slice:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ђђ:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
Ђ
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_221683

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
»
^
B__inference_lambda_layer_call_and_return_conditional_losses_222041

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2€
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
strided_slicet
IdentityIdentitystrided_slice:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ђђ:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
а
C
'__inference_lambda_layer_call_fn_222054

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2203972
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ђђ:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
в
q
G__inference_concatenate_layer_call_and_return_conditional_losses_220183

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisА
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Џ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Џ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€:€€€€€€€€€Ў:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€Ў
 
_user_specified_nameinputs
к
J
.__inference_up_sampling2d_layer_call_fn_221832

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_2202522
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
—D
ј
__inference__traced_save_222212
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename√
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*’
valueЋB»#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesќ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices®
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*Є
_input_shapes¶
£: :Ц:Ц:ЦЦ:Ц:ЦЦ:Ц:ЦЦ:Ц:
ЏЎ:Ў:ЦЦ:Ц:ЦЦ:Ц:ЦЦ:Ц:ЦЦ:Ц:Ц:: : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:Ц:!

_output_shapes	
:Ц:.*
(
_output_shapes
:ЦЦ:!

_output_shapes	
:Ц:.*
(
_output_shapes
:ЦЦ:!

_output_shapes	
:Ц:.*
(
_output_shapes
:ЦЦ:!

_output_shapes	
:Ц:&	"
 
_output_shapes
:
ЏЎ:!


_output_shapes	
:Ў:.*
(
_output_shapes
:ЦЦ:!

_output_shapes	
:Ц:.*
(
_output_shapes
:ЦЦ:!

_output_shapes	
:Ц:.*
(
_output_shapes
:ЦЦ:!

_output_shapes	
:Ц:.*
(
_output_shapes
:ЦЦ:!

_output_shapes	
:Ц:-)
'
_output_shapes
:Ц: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: 
¶
В
D__inference_conv2d_3_layer_call_and_return_conditional_losses_220148

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-220141*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
Ђ
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_221638

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
„
L
0__inference_max_pooling2d_1_layer_call_fn_221603

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2198382
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
о
L
0__inference_max_pooling2d_1_layer_call_fn_221608

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2201022
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€<<Ц:X T
0
_output_shapes
:€€€€€€€€€<<Ц
 
_user_specified_nameinputs
э
u
I__inference_masking_layer_layer_call_and_return_conditional_losses_222027
inputs_0
inputs_1
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xg
subSubsub/x:output:0inputs_1*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
suba
mulMulinputs_0inputs_1*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
mulW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
mul_1/yl
mul_1Mulsub:z:0mul_1/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
mul_1c
addAddV2mul:z:0	mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::€€€€€€€€€ђђ:€€€€€€€€€ђђ:[ W
1
_output_shapes
:€€€€€€€€€ђђ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:€€€€€€€€€ђђ
"
_user_specified_name
inputs/1
я
Е
&__inference_model_layer_call_fn_221470
inputs_0
inputs_1"
unknown:Ц
	unknown_0:	Ц%
	unknown_1:ЦЦ
	unknown_2:	Ц%
	unknown_3:ЦЦ
	unknown_4:	Ц%
	unknown_5:ЦЦ
	unknown_6:	Ц
	unknown_7:
ЏЎ
	unknown_8:	Ў%
	unknown_9:ЦЦ

unknown_10:	Ц&

unknown_11:ЦЦ

unknown_12:	Ц&

unknown_13:ЦЦ

unknown_14:	Ц&

unknown_15:ЦЦ

unknown_16:	Ц%

unknown_17:Ц

unknown_18:
identity

identity_1ИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::€€€€€€€€€ђђ:€€€€€€€€€ђђ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2204012
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

IdentityЙ

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:€€€€€€€€€ђђ:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:€€€€€€€€€ђђ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
¶
В
D__inference_conv2d_2_layer_call_and_return_conditional_losses_221624

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221617*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
°
°
)__inference_conv2d_5_layer_call_fn_221857

inputs#
unknown:ЦЦ
	unknown_0:	Ц
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_2202702
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
¶
В
D__inference_conv2d_6_layer_call_and_return_conditional_losses_221903

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221896*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
ъk
ƒ	
A__inference_model_layer_call_and_return_conditional_losses_220781

inputs
inputs_1(
conv2d_220711:Ц
conv2d_220713:	Ц+
conv2d_1_220717:ЦЦ
conv2d_1_220719:	Ц+
conv2d_2_220723:ЦЦ
conv2d_2_220725:	Ц+
conv2d_3_220729:ЦЦ
conv2d_3_220731:	Ц 
dense_220738:
ЏЎ
dense_220740:	Ў+
conv2d_4_220744:ЦЦ
conv2d_4_220746:	Ц+
conv2d_5_220750:ЦЦ
conv2d_5_220752:	Ц+
conv2d_6_220756:ЦЦ
conv2d_6_220758:	Ц+
conv2d_7_220762:ЦЦ
conv2d_7_220764:	Ц*
conv2d_8_220770:Ц
conv2d_8_220772:
identity

identity_1ИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallҐdense/StatefulPartitionedCallХ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_220711conv2d_220713*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:€€€€€€€€€ђђЦ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_2200642 
conv2d/StatefulPartitionedCallН
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_2200742
max_pooling2d/PartitionedCallљ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_220717conv2d_1_220719*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2200922"
 conv2d_1/StatefulPartitionedCallХ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2201022!
max_pooling2d_1/PartitionedCallњ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_220723conv2d_2_220725*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2201202"
 conv2d_2/StatefulPartitionedCallХ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2201302!
max_pooling2d_2/PartitionedCallњ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_220729conv2d_3_220731*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2201482"
 conv2d_3/StatefulPartitionedCallХ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2201582!
max_pooling2d_3/PartitionedCallў
flatten_1/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2201662
flatten_1/PartitionedCallф
flatten/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ў* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2201742
flatten/PartitionedCallЭ
concatenate/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Џ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2201832
concatenate/PartitionedCall§
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_220738dense_220740*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ў*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2202012
dense/StatefulPartitionedCallъ
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_2202212
reshape/PartitionedCallЈ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_4_220744conv2d_4_220746*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_2202392"
 conv2d_4/StatefulPartitionedCallП
up_sampling2d/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_2202522
up_sampling2d/PartitionedCallљ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_5_220750conv2d_5_220752*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_2202702"
 conv2d_5/StatefulPartitionedCallХ
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_2202832!
up_sampling2d_1/PartitionedCallњ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_6_220756conv2d_6_220758*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2203012"
 conv2d_6/StatefulPartitionedCallХ
up_sampling2d_2/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2203142!
up_sampling2d_2/PartitionedCallњ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_7_220762conv2d_7_220764*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2203322"
 conv2d_7/StatefulPartitionedCall{
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
tf.math.greater/Greater/y•
tf.math.greater/GreaterGreaterinputs"tf.math.greater/Greater/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
tf.math.greater/GreaterЧ
up_sampling2d_3/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:€€€€€€€€€ђђЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2203472!
up_sampling2d_3/PartitionedCallј
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_8_220770conv2d_8_220772*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2203592"
 conv2d_8/StatefulPartitionedCallМ
tf.cast/CastCasttf.math.greater/Greater:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€ђђ2
tf.cast/Cast£
masking_layer/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0tf.cast/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_masking_layer_layer_call_and_return_conditional_losses_2203772
masking_layer/PartitionedCallю
lambda_1/PartitionedCallPartitionedCall&masking_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_2204812
lambda_1/PartitionedCallш
lambda/PartitionedCallPartitionedCall&masking_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2204622
lambda/PartitionedCallД
IdentityIdentitylambda/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

IdentityК

Identity_1Identity!lambda_1/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity_1І
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:€€€€€€€€€ђђ:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
£–
л
A__inference_model_layer_call_and_return_conditional_losses_221422
inputs_0
inputs_1@
%conv2d_conv2d_readvariableop_resource:Ц5
&conv2d_biasadd_readvariableop_resource:	ЦC
'conv2d_1_conv2d_readvariableop_resource:ЦЦ7
(conv2d_1_biasadd_readvariableop_resource:	ЦC
'conv2d_2_conv2d_readvariableop_resource:ЦЦ7
(conv2d_2_biasadd_readvariableop_resource:	ЦC
'conv2d_3_conv2d_readvariableop_resource:ЦЦ7
(conv2d_3_biasadd_readvariableop_resource:	Ц8
$dense_matmul_readvariableop_resource:
ЏЎ4
%dense_biasadd_readvariableop_resource:	ЎC
'conv2d_4_conv2d_readvariableop_resource:ЦЦ7
(conv2d_4_biasadd_readvariableop_resource:	ЦC
'conv2d_5_conv2d_readvariableop_resource:ЦЦ7
(conv2d_5_biasadd_readvariableop_resource:	ЦC
'conv2d_6_conv2d_readvariableop_resource:ЦЦ7
(conv2d_6_biasadd_readvariableop_resource:	ЦC
'conv2d_7_conv2d_readvariableop_resource:ЦЦ7
(conv2d_7_biasadd_readvariableop_resource:	ЦB
'conv2d_8_conv2d_readvariableop_resource:Ц6
(conv2d_8_biasadd_readvariableop_resource:
identity

identity_1ИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOpҐconv2d_4/BiasAdd/ReadVariableOpҐconv2d_4/Conv2D/ReadVariableOpҐconv2d_5/BiasAdd/ReadVariableOpҐconv2d_5/Conv2D/ReadVariableOpҐconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpҐconv2d_7/BiasAdd/ReadVariableOpҐconv2d_7/Conv2D/ReadVariableOpҐconv2d_8/BiasAdd/ReadVariableOpҐconv2d_8/Conv2D/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpЂ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:Ц*
dtype02
conv2d/Conv2D/ReadVariableOpљ
conv2d/Conv2DConv2Dinputs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ*
paddingSAME*
strides
2
conv2d/Conv2DҐ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
conv2d/BiasAdd/ReadVariableOpІ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2
conv2d/BiasAddБ
conv2d/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2
conv2d/SigmoidЙ

conv2d/mulMulconv2d/BiasAdd:output:0conv2d/Sigmoid:y:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2

conv2d/mul{
conv2d/IdentityIdentityconv2d/mul:z:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2
conv2d/Identityз
conv2d/IdentityN	IdentityNconv2d/mul:z:0conv2d/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221259*P
_output_shapes>
<:€€€€€€€€€ђђЦ:€€€€€€€€€ђђЦ2
conv2d/IdentityN¬
max_pooling2d/MaxPoolMaxPoolconv2d/IdentityN:output:0*0
_output_shapes
:€€€€€€€€€<<Ц*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool≤
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_1/Conv2D/ReadVariableOp„
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
paddingSAME*
strides
2
conv2d_1/Conv2D®
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp≠
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_1/BiasAddЕ
conv2d_1/SigmoidSigmoidconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_1/SigmoidП
conv2d_1/mulMulconv2d_1/BiasAdd:output:0conv2d_1/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_1/mul
conv2d_1/IdentityIdentityconv2d_1/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_1/Identityл
conv2d_1/IdentityN	IdentityNconv2d_1/mul:z:0conv2d_1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221272*L
_output_shapes:
8:€€€€€€€€€<<Ц:€€€€€€€€€<<Ц2
conv2d_1/IdentityN»
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/IdentityN:output:0*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool≤
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_2/Conv2D/ReadVariableOpў
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
conv2d_2/Conv2D®
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp≠
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_2/BiasAddЕ
conv2d_2/SigmoidSigmoidconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_2/SigmoidП
conv2d_2/mulMulconv2d_2/BiasAdd:output:0conv2d_2/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_2/mul
conv2d_2/IdentityIdentityconv2d_2/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_2/Identityл
conv2d_2/IdentityN	IdentityNconv2d_2/mul:z:0conv2d_2/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221285*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
conv2d_2/IdentityN»
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/IdentityN:output:0*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool≤
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_3/Conv2D/ReadVariableOpў
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
conv2d_3/Conv2D®
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp≠
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_3/BiasAddЕ
conv2d_3/SigmoidSigmoidconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_3/SigmoidП
conv2d_3/mulMulconv2d_3/BiasAdd:output:0conv2d_3/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_3/mul
conv2d_3/IdentityIdentityconv2d_3/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_3/Identityл
conv2d_3/IdentityN	IdentityNconv2d_3/mul:z:0conv2d_3/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221298*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
conv2d_3/IdentityN»
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/IdentityN:output:0*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPools
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten_1/ConstЗ
flatten_1/ReshapeReshapeinputs_1flatten_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€X  2
flatten/ConstЪ
flatten/ReshapeReshape max_pooling2d_3/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
flatten/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis»
concatenate/concatConcatV2flatten_1/Reshape:output:0flatten/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Џ2
concatenate/concat°
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ЏЎ*
dtype02
dense/MatMul/ReadVariableOpЫ
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:Ў*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
dense/BiasAddt
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
dense/Sigmoid{
	dense/mulMuldense/BiasAdd:output:0dense/Sigmoid:y:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
	dense/muln
dense/IdentityIdentitydense/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
dense/Identityѕ
dense/IdentityN	IdentityNdense/mul:z:0dense/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221317*<
_output_shapes*
(:€€€€€€€€€Ў:€€€€€€€€€Ў2
dense/IdentityNf
reshape/ShapeShapedense/IdentityN:output:0*
T0*
_output_shapes
:2
reshape/ShapeД
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackИ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1И
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2Т
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :Ц2
reshape/Reshape/shape/3к
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeҐ
reshape/ReshapeReshapedense/IdentityN:output:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
reshape/Reshape≤
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_4/Conv2D/ReadVariableOp—
conv2d_4/Conv2DConv2Dreshape/Reshape:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
conv2d_4/Conv2D®
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp≠
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_4/BiasAddЕ
conv2d_4/SigmoidSigmoidconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_4/SigmoidП
conv2d_4/mulMulconv2d_4/BiasAdd:output:0conv2d_4/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_4/mul
conv2d_4/IdentityIdentityconv2d_4/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_4/Identityл
conv2d_4/IdentityN	IdentityNconv2d_4/mul:z:0conv2d_4/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221339*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
conv2d_4/IdentityN{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1Р
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mulъ
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_4/IdentityN:output:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor≤
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_5/Conv2D/ReadVariableOpф
conv2d_5/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
conv2d_5/Conv2D®
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp≠
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_5/BiasAddЕ
conv2d_5/SigmoidSigmoidconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_5/SigmoidП
conv2d_5/mulMulconv2d_5/BiasAdd:output:0conv2d_5/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_5/mul
conv2d_5/IdentityIdentityconv2d_5/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_5/Identityл
conv2d_5/IdentityN	IdentityNconv2d_5/mul:z:0conv2d_5/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221355*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
conv2d_5/IdentityN
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/ConstГ
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1Ш
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mulА
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_5/IdentityN:output:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor≤
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_6/Conv2D/ReadVariableOpц
conv2d_6/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
conv2d_6/Conv2D®
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp≠
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_6/BiasAddЕ
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_6/SigmoidП
conv2d_6/mulMulconv2d_6/BiasAdd:output:0conv2d_6/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_6/mul
conv2d_6/IdentityIdentityconv2d_6/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_6/Identityл
conv2d_6/IdentityN	IdentityNconv2d_6/mul:z:0conv2d_6/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221371*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
conv2d_6/IdentityN
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/ConstГ
up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const_1Ш
up_sampling2d_2/mulMulup_sampling2d_2/Const:output:0 up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mulА
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_6/IdentityN:output:0up_sampling2d_2/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor≤
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_7/Conv2D/ReadVariableOpц
conv2d_7/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
paddingSAME*
strides
2
conv2d_7/Conv2D®
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp≠
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_7/BiasAddЕ
conv2d_7/SigmoidSigmoidconv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_7/SigmoidП
conv2d_7/mulMulconv2d_7/BiasAdd:output:0conv2d_7/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_7/mul
conv2d_7/IdentityIdentityconv2d_7/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_7/Identityл
conv2d_7/IdentityN	IdentityNconv2d_7/mul:z:0conv2d_7/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221387*L
_output_shapes:
8:€€€€€€€€€<<Ц:€€€€€€€€€<<Ц2
conv2d_7/IdentityN{
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
tf.math.greater/Greater/yІ
tf.math.greater/GreaterGreaterinputs_0"tf.math.greater/Greater/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
tf.math.greater/Greater
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"<   <   2
up_sampling2d_3/ConstГ
up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const_1Ш
up_sampling2d_3/mulMulup_sampling2d_3/Const:output:0 up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mulВ
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_7/IdentityN:output:0up_sampling2d_3/mul:z:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor±
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*'
_output_shapes
:Ц*
dtype02 
conv2d_8/Conv2D/ReadVariableOpч
conv2d_8/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*
paddingSAME*
strides
2
conv2d_8/Conv2DІ
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOpЃ
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
conv2d_8/BiasAddМ
tf.cast/CastCasttf.math.greater/Greater:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€ђђ2
tf.cast/Casto
masking_layer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
masking_layer/sub/xЩ
masking_layer/subSubmasking_layer/sub/x:output:0tf.cast/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
masking_layer/subЦ
masking_layer/mulMulconv2d_8/BiasAdd:output:0tf.cast/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
masking_layer/muls
masking_layer/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
masking_layer/mul_1/y§
masking_layer/mul_1Mulmasking_layer/sub:z:0masking_layer/mul_1/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
masking_layer/mul_1Ы
masking_layer/addAddV2masking_layer/mul:z:0masking_layer/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
masking_layer/addХ
lambda_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
lambda_1/strided_slice/stackЩ
lambda_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2 
lambda_1/strided_slice/stack_1Щ
lambda_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_1/strided_slice/stack_2ї
lambda_1/strided_sliceStridedSlicemasking_layer/add:z:0%lambda_1/strided_slice/stack:output:0'lambda_1/strided_slice/stack_1:output:0'lambda_1/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
lambda_1/strided_sliceС
lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda/strided_slice/stackХ
lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
lambda/strided_slice/stack_1Х
lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
lambda/strided_slice/stack_2±
lambda/strided_sliceStridedSlicemasking_layer/add:z:0#lambda/strided_slice/stack:output:0%lambda/strided_slice/stack_1:output:0%lambda/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
lambda/strided_sliceВ
IdentityIdentitylambda/strided_slice:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

IdentityИ

Identity_1Identitylambda_1/strided_slice:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity_1в
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:€€€€€€€€€ђђ:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:[ W
1
_output_shapes
:€€€€€€€€€ђђ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
і
€
B__inference_conv2d_layer_call_and_return_conditional_losses_220064

inputs9
conv2d_readvariableop_resource:Ц.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:Ц*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЛ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2	
BiasAddl
SigmoidSigmoidBiasAdd:output:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2	
Sigmoidm
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2
mulf
IdentityIdentitymul:z:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2

IdentityЋ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-220057*P
_output_shapes>
<:€€€€€€€€€ђђЦ:€€€€€€€€€ђђЦ2
	IdentityN|

Identity_1IdentityIdentityN:output:0^NoOp*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€ђђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
©
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_221548

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ђ
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_219838

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
”
J
.__inference_max_pooling2d_layer_call_fn_221558

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_2198162
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ќ
D
(__inference_reshape_layer_call_fn_221777

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_2202212
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€Ў:P L
(
_output_shapes
:€€€€€€€€€Ў
 
_user_specified_nameinputs
і
€
B__inference_conv2d_layer_call_and_return_conditional_losses_221534

inputs9
conv2d_readvariableop_resource:Ц.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:Ц*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЛ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2	
BiasAddl
SigmoidSigmoidBiasAdd:output:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2	
Sigmoidm
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2
mulf
IdentityIdentitymul:z:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2

IdentityЋ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221527*P
_output_shapes>
<:€€€€€€€€€ђђЦ:€€€€€€€€€ђђЦ2
	IdentityN|

Identity_1IdentityIdentityN:output:0^NoOp*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€ђђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
£–
л
A__inference_model_layer_call_and_return_conditional_losses_221249
inputs_0
inputs_1@
%conv2d_conv2d_readvariableop_resource:Ц5
&conv2d_biasadd_readvariableop_resource:	ЦC
'conv2d_1_conv2d_readvariableop_resource:ЦЦ7
(conv2d_1_biasadd_readvariableop_resource:	ЦC
'conv2d_2_conv2d_readvariableop_resource:ЦЦ7
(conv2d_2_biasadd_readvariableop_resource:	ЦC
'conv2d_3_conv2d_readvariableop_resource:ЦЦ7
(conv2d_3_biasadd_readvariableop_resource:	Ц8
$dense_matmul_readvariableop_resource:
ЏЎ4
%dense_biasadd_readvariableop_resource:	ЎC
'conv2d_4_conv2d_readvariableop_resource:ЦЦ7
(conv2d_4_biasadd_readvariableop_resource:	ЦC
'conv2d_5_conv2d_readvariableop_resource:ЦЦ7
(conv2d_5_biasadd_readvariableop_resource:	ЦC
'conv2d_6_conv2d_readvariableop_resource:ЦЦ7
(conv2d_6_biasadd_readvariableop_resource:	ЦC
'conv2d_7_conv2d_readvariableop_resource:ЦЦ7
(conv2d_7_biasadd_readvariableop_resource:	ЦB
'conv2d_8_conv2d_readvariableop_resource:Ц6
(conv2d_8_biasadd_readvariableop_resource:
identity

identity_1ИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOpҐconv2d_4/BiasAdd/ReadVariableOpҐconv2d_4/Conv2D/ReadVariableOpҐconv2d_5/BiasAdd/ReadVariableOpҐconv2d_5/Conv2D/ReadVariableOpҐconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpҐconv2d_7/BiasAdd/ReadVariableOpҐconv2d_7/Conv2D/ReadVariableOpҐconv2d_8/BiasAdd/ReadVariableOpҐconv2d_8/Conv2D/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpЂ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:Ц*
dtype02
conv2d/Conv2D/ReadVariableOpљ
conv2d/Conv2DConv2Dinputs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ*
paddingSAME*
strides
2
conv2d/Conv2DҐ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
conv2d/BiasAdd/ReadVariableOpІ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2
conv2d/BiasAddБ
conv2d/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2
conv2d/SigmoidЙ

conv2d/mulMulconv2d/BiasAdd:output:0conv2d/Sigmoid:y:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2

conv2d/mul{
conv2d/IdentityIdentityconv2d/mul:z:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2
conv2d/Identityз
conv2d/IdentityN	IdentityNconv2d/mul:z:0conv2d/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221086*P
_output_shapes>
<:€€€€€€€€€ђђЦ:€€€€€€€€€ђђЦ2
conv2d/IdentityN¬
max_pooling2d/MaxPoolMaxPoolconv2d/IdentityN:output:0*0
_output_shapes
:€€€€€€€€€<<Ц*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool≤
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_1/Conv2D/ReadVariableOp„
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
paddingSAME*
strides
2
conv2d_1/Conv2D®
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp≠
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_1/BiasAddЕ
conv2d_1/SigmoidSigmoidconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_1/SigmoidП
conv2d_1/mulMulconv2d_1/BiasAdd:output:0conv2d_1/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_1/mul
conv2d_1/IdentityIdentityconv2d_1/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_1/Identityл
conv2d_1/IdentityN	IdentityNconv2d_1/mul:z:0conv2d_1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221099*L
_output_shapes:
8:€€€€€€€€€<<Ц:€€€€€€€€€<<Ц2
conv2d_1/IdentityN»
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/IdentityN:output:0*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool≤
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_2/Conv2D/ReadVariableOpў
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
conv2d_2/Conv2D®
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp≠
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_2/BiasAddЕ
conv2d_2/SigmoidSigmoidconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_2/SigmoidП
conv2d_2/mulMulconv2d_2/BiasAdd:output:0conv2d_2/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_2/mul
conv2d_2/IdentityIdentityconv2d_2/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_2/Identityл
conv2d_2/IdentityN	IdentityNconv2d_2/mul:z:0conv2d_2/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221112*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
conv2d_2/IdentityN»
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/IdentityN:output:0*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool≤
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_3/Conv2D/ReadVariableOpў
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
conv2d_3/Conv2D®
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp≠
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_3/BiasAddЕ
conv2d_3/SigmoidSigmoidconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_3/SigmoidП
conv2d_3/mulMulconv2d_3/BiasAdd:output:0conv2d_3/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_3/mul
conv2d_3/IdentityIdentityconv2d_3/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_3/Identityл
conv2d_3/IdentityN	IdentityNconv2d_3/mul:z:0conv2d_3/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221125*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
conv2d_3/IdentityN»
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/IdentityN:output:0*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPools
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten_1/ConstЗ
flatten_1/ReshapeReshapeinputs_1flatten_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€X  2
flatten/ConstЪ
flatten/ReshapeReshape max_pooling2d_3/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
flatten/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis»
concatenate/concatConcatV2flatten_1/Reshape:output:0flatten/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Џ2
concatenate/concat°
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ЏЎ*
dtype02
dense/MatMul/ReadVariableOpЫ
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:Ў*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
dense/BiasAddt
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
dense/Sigmoid{
	dense/mulMuldense/BiasAdd:output:0dense/Sigmoid:y:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
	dense/muln
dense/IdentityIdentitydense/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
dense/Identityѕ
dense/IdentityN	IdentityNdense/mul:z:0dense/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221144*<
_output_shapes*
(:€€€€€€€€€Ў:€€€€€€€€€Ў2
dense/IdentityNf
reshape/ShapeShapedense/IdentityN:output:0*
T0*
_output_shapes
:2
reshape/ShapeД
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackИ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1И
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2Т
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :Ц2
reshape/Reshape/shape/3к
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeҐ
reshape/ReshapeReshapedense/IdentityN:output:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
reshape/Reshape≤
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_4/Conv2D/ReadVariableOp—
conv2d_4/Conv2DConv2Dreshape/Reshape:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
conv2d_4/Conv2D®
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp≠
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_4/BiasAddЕ
conv2d_4/SigmoidSigmoidconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_4/SigmoidП
conv2d_4/mulMulconv2d_4/BiasAdd:output:0conv2d_4/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_4/mul
conv2d_4/IdentityIdentityconv2d_4/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_4/Identityл
conv2d_4/IdentityN	IdentityNconv2d_4/mul:z:0conv2d_4/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221166*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
conv2d_4/IdentityN{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1Р
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mulъ
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_4/IdentityN:output:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor≤
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_5/Conv2D/ReadVariableOpф
conv2d_5/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
conv2d_5/Conv2D®
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp≠
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_5/BiasAddЕ
conv2d_5/SigmoidSigmoidconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_5/SigmoidП
conv2d_5/mulMulconv2d_5/BiasAdd:output:0conv2d_5/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_5/mul
conv2d_5/IdentityIdentityconv2d_5/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_5/Identityл
conv2d_5/IdentityN	IdentityNconv2d_5/mul:z:0conv2d_5/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221182*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
conv2d_5/IdentityN
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/ConstГ
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1Ш
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mulА
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_5/IdentityN:output:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor≤
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_6/Conv2D/ReadVariableOpц
conv2d_6/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
conv2d_6/Conv2D®
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp≠
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_6/BiasAddЕ
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_6/SigmoidП
conv2d_6/mulMulconv2d_6/BiasAdd:output:0conv2d_6/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_6/mul
conv2d_6/IdentityIdentityconv2d_6/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
conv2d_6/Identityл
conv2d_6/IdentityN	IdentityNconv2d_6/mul:z:0conv2d_6/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221198*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
conv2d_6/IdentityN
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/ConstГ
up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const_1Ш
up_sampling2d_2/mulMulup_sampling2d_2/Const:output:0 up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mulА
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_6/IdentityN:output:0up_sampling2d_2/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor≤
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02 
conv2d_7/Conv2D/ReadVariableOpц
conv2d_7/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
paddingSAME*
strides
2
conv2d_7/Conv2D®
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp≠
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_7/BiasAddЕ
conv2d_7/SigmoidSigmoidconv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_7/SigmoidП
conv2d_7/mulMulconv2d_7/BiasAdd:output:0conv2d_7/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_7/mul
conv2d_7/IdentityIdentityconv2d_7/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
conv2d_7/Identityл
conv2d_7/IdentityN	IdentityNconv2d_7/mul:z:0conv2d_7/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221214*L
_output_shapes:
8:€€€€€€€€€<<Ц:€€€€€€€€€<<Ц2
conv2d_7/IdentityN{
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
tf.math.greater/Greater/yІ
tf.math.greater/GreaterGreaterinputs_0"tf.math.greater/Greater/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
tf.math.greater/Greater
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"<   <   2
up_sampling2d_3/ConstГ
up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const_1Ш
up_sampling2d_3/mulMulup_sampling2d_3/Const:output:0 up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mulВ
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_7/IdentityN:output:0up_sampling2d_3/mul:z:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor±
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*'
_output_shapes
:Ц*
dtype02 
conv2d_8/Conv2D/ReadVariableOpч
conv2d_8/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*
paddingSAME*
strides
2
conv2d_8/Conv2DІ
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOpЃ
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
conv2d_8/BiasAddМ
tf.cast/CastCasttf.math.greater/Greater:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€ђђ2
tf.cast/Casto
masking_layer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
masking_layer/sub/xЩ
masking_layer/subSubmasking_layer/sub/x:output:0tf.cast/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
masking_layer/subЦ
masking_layer/mulMulconv2d_8/BiasAdd:output:0tf.cast/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
masking_layer/muls
masking_layer/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
masking_layer/mul_1/y§
masking_layer/mul_1Mulmasking_layer/sub:z:0masking_layer/mul_1/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
masking_layer/mul_1Ы
masking_layer/addAddV2masking_layer/mul:z:0masking_layer/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
masking_layer/addХ
lambda_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
lambda_1/strided_slice/stackЩ
lambda_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2 
lambda_1/strided_slice/stack_1Щ
lambda_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_1/strided_slice/stack_2ї
lambda_1/strided_sliceStridedSlicemasking_layer/add:z:0%lambda_1/strided_slice/stack:output:0'lambda_1/strided_slice/stack_1:output:0'lambda_1/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
lambda_1/strided_sliceС
lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda/strided_slice/stackХ
lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
lambda/strided_slice/stack_1Х
lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
lambda/strided_slice/stack_2±
lambda/strided_sliceStridedSlicemasking_layer/add:z:0#lambda/strided_slice/stack:output:0%lambda/strided_slice/stack_1:output:0%lambda/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
lambda/strided_sliceВ
IdentityIdentitylambda/strided_slice:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

IdentityИ

Identity_1Identitylambda_1/strided_slice:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity_1в
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:€€€€€€€€€ђђ:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:[ W
1
_output_shapes
:€€€€€€€€€ђђ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
¬
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_220102

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€<<Ц:X T
0
_output_shapes
:€€€€€€€€€<<Ц
 
_user_specified_nameinputs
≤
g
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_219947

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ќ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul’
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(2
resize/ResizeNearestNeighbor§
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
„
L
0__inference_max_pooling2d_2_layer_call_fn_221648

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2198602
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ђ
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_219860

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
З
Z
.__inference_masking_layer_layer_call_fn_222033
inputs_0
inputs_1
identityё
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_masking_layer_layer_call_and_return_conditional_losses_2203772
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::€€€€€€€€€ђђ:€€€€€€€€€ђђ:[ W
1
_output_shapes
:€€€€€€€€€ђђ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:€€€€€€€€€ђђ
"
_user_specified_name
inputs/1
»
^
B__inference_lambda_layer_call_and_return_conditional_losses_222049

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2€
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
strided_slicet
IdentityIdentitystrided_slice:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ђђ:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
¶
В
D__inference_conv2d_5_layer_call_and_return_conditional_losses_220270

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-220263*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
»
^
B__inference_lambda_layer_call_and_return_conditional_losses_220462

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2€
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
strided_slicet
IdentityIdentitystrided_slice:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ђђ:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
о
L
0__inference_max_pooling2d_2_layer_call_fn_221653

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2201302
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
≤
g
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_221924

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ќ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul’
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(2
resize/ResizeNearestNeighbor§
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
®
g
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_220283

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Constc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2	
Const_1X
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:2
mulї
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
half_pixel_centers(2
resize/ResizeNearestNeighborК
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
ў
Г
&__inference_model_layer_call_fn_220874
input_1
input_2"
unknown:Ц
	unknown_0:	Ц%
	unknown_1:ЦЦ
	unknown_2:	Ц%
	unknown_3:ЦЦ
	unknown_4:	Ц%
	unknown_5:ЦЦ
	unknown_6:	Ц
	unknown_7:
ЏЎ
	unknown_8:	Ў%
	unknown_9:ЦЦ

unknown_10:	Ц&

unknown_11:ЦЦ

unknown_12:	Ц&

unknown_13:ЦЦ

unknown_14:	Ц&

unknown_15:ЦЦ

unknown_16:	Ц%

unknown_17:Ц

unknown_18:
identity

identity_1ИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::€€€€€€€€€ђђ:€€€€€€€€€ђђ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2207812
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

IdentityЙ

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:€€€€€€€€€ђђ:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€ђђ
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_2
’
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_220166

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
а
C
'__inference_lambda_layer_call_fn_222059

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2204622
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ђђ:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
¬
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_221643

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
ъk
ƒ	
A__inference_model_layer_call_and_return_conditional_losses_220401

inputs
inputs_1(
conv2d_220065:Ц
conv2d_220067:	Ц+
conv2d_1_220093:ЦЦ
conv2d_1_220095:	Ц+
conv2d_2_220121:ЦЦ
conv2d_2_220123:	Ц+
conv2d_3_220149:ЦЦ
conv2d_3_220151:	Ц 
dense_220202:
ЏЎ
dense_220204:	Ў+
conv2d_4_220240:ЦЦ
conv2d_4_220242:	Ц+
conv2d_5_220271:ЦЦ
conv2d_5_220273:	Ц+
conv2d_6_220302:ЦЦ
conv2d_6_220304:	Ц+
conv2d_7_220333:ЦЦ
conv2d_7_220335:	Ц*
conv2d_8_220360:Ц
conv2d_8_220362:
identity

identity_1ИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallҐdense/StatefulPartitionedCallХ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_220065conv2d_220067*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:€€€€€€€€€ђђЦ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_2200642 
conv2d/StatefulPartitionedCallН
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_2200742
max_pooling2d/PartitionedCallљ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_220093conv2d_1_220095*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2200922"
 conv2d_1/StatefulPartitionedCallХ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2201022!
max_pooling2d_1/PartitionedCallњ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_220121conv2d_2_220123*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2201202"
 conv2d_2/StatefulPartitionedCallХ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2201302!
max_pooling2d_2/PartitionedCallњ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_220149conv2d_3_220151*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2201482"
 conv2d_3/StatefulPartitionedCallХ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2201582!
max_pooling2d_3/PartitionedCallў
flatten_1/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2201662
flatten_1/PartitionedCallф
flatten/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ў* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2201742
flatten/PartitionedCallЭ
concatenate/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Џ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2201832
concatenate/PartitionedCall§
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_220202dense_220204*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ў*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2202012
dense/StatefulPartitionedCallъ
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_2202212
reshape/PartitionedCallЈ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_4_220240conv2d_4_220242*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_2202392"
 conv2d_4/StatefulPartitionedCallП
up_sampling2d/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_2202522
up_sampling2d/PartitionedCallљ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_5_220271conv2d_5_220273*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_2202702"
 conv2d_5/StatefulPartitionedCallХ
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_2202832!
up_sampling2d_1/PartitionedCallњ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_6_220302conv2d_6_220304*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2203012"
 conv2d_6/StatefulPartitionedCallХ
up_sampling2d_2/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2203142!
up_sampling2d_2/PartitionedCallњ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_7_220333conv2d_7_220335*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2203322"
 conv2d_7/StatefulPartitionedCall{
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
tf.math.greater/Greater/y•
tf.math.greater/GreaterGreaterinputs"tf.math.greater/Greater/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
tf.math.greater/GreaterЧ
up_sampling2d_3/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:€€€€€€€€€ђђЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2203472!
up_sampling2d_3/PartitionedCallј
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_8_220360conv2d_8_220362*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2203592"
 conv2d_8/StatefulPartitionedCallМ
tf.cast/CastCasttf.math.greater/Greater:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€ђђ2
tf.cast/Cast£
masking_layer/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0tf.cast/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_masking_layer_layer_call_and_return_conditional_losses_2203772
masking_layer/PartitionedCallю
lambda_1/PartitionedCallPartitionedCall&masking_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_2203872
lambda_1/PartitionedCallш
lambda/PartitionedCallPartitionedCall&masking_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2203972
lambda/PartitionedCallД
IdentityIdentitylambda/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

IdentityК

Identity_1Identity!lambda_1/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity_1І
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:€€€€€€€€€ђђ:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
о
L
0__inference_max_pooling2d_3_layer_call_fn_221698

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2201582
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
°
°
)__inference_conv2d_6_layer_call_fn_221912

inputs#
unknown:ЦЦ
	unknown_0:	Ц
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2203012
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
®
g
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_220314

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Constc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2	
Const_1X
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:2
mulї
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
half_pixel_centers(2
resize/ResizeNearestNeighborК
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
з
_
C__inference_flatten_layer_call_and_return_conditional_losses_220174

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€X  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
з
_
C__inference_flatten_layer_call_and_return_conditional_losses_221715

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€X  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
У
_
C__inference_reshape_layer_call_and_return_conditional_losses_221772

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :Ц2
Reshape/shape/3Ї
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€Ў:P L
(
_output_shapes
:€€€€€€€€€Ў
 
_user_specified_nameinputs
 
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_222067

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2€
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
strided_slicet
IdentityIdentitystrided_slice:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ђђ:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
У
_
C__inference_reshape_layer_call_and_return_conditional_losses_220221

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :Ц2
Reshape/shape/3Ї
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€Ў:P L
(
_output_shapes
:€€€€€€€€€Ў
 
_user_specified_nameinputs
„
L
0__inference_up_sampling2d_3_layer_call_fn_221992

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2200192
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
д
E
)__inference_lambda_1_layer_call_fn_222085

inputs
identityћ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_2204812
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ђђ:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
ђ
g
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_220347

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"<   <   2
Constc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2	
Const_1X
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:2
mulљ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ*
half_pixel_centers(2
resize/ResizeNearestNeighborМ
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€<<Ц:X T
0
_output_shapes
:€€€€€€€€€<<Ц
 
_user_specified_nameinputs
т
L
0__inference_up_sampling2d_3_layer_call_fn_221997

inputs
identity‘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:€€€€€€€€€ђђЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2203472
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€<<Ц:X T
0
_output_shapes
:€€€€€€€€€<<Ц
 
_user_specified_nameinputs
Ћ
X
,__inference_concatenate_layer_call_fn_221733
inputs_0
inputs_1
identity”
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Џ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2201832
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€Џ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€:€€€€€€€€€Ў:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:€€€€€€€€€Ў
"
_user_specified_name
inputs/1
о
L
0__inference_up_sampling2d_2_layer_call_fn_221942

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2203142
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
¶
В
D__inference_conv2d_4_layer_call_and_return_conditional_losses_221793

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221786*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
°
°
)__inference_conv2d_1_layer_call_fn_221588

inputs#
unknown:ЦЦ
	unknown_0:	Ц
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2200922
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€<<Ц: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€<<Ц
 
_user_specified_nameinputs
¶
В
D__inference_conv2d_1_layer_call_and_return_conditional_losses_221579

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221572*L
_output_shapes:
8:€€€€€€€€€<<Ц:€€€€€€€€€<<Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€<<Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€<<Ц
 
_user_specified_nameinputs
д
E
)__inference_lambda_1_layer_call_fn_222080

inputs
identityћ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_2203872
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ђђ:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
¶
e
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_220252

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Constc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2	
Const_1X
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:2
mulї
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
half_pixel_centers(2
resize/ResizeNearestNeighborК
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
¶
В
D__inference_conv2d_3_layer_call_and_return_conditional_losses_221669

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221662*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
Ђ
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_219882

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
e
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_219911

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ќ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul’
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(2
resize/ResizeNearestNeighbor§
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
¶
В
D__inference_conv2d_5_layer_call_and_return_conditional_losses_221848

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221841*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
°
°
)__inference_conv2d_2_layer_call_fn_221633

inputs#
unknown:ЦЦ
	unknown_0:	Ц
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2201202
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
®
g
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_221932

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Constc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2	
Const_1X
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:2
mulї
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
half_pixel_centers(2
resize/ResizeNearestNeighborК
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
„
L
0__inference_up_sampling2d_2_layer_call_fn_221937

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2199832
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
°
°
)__inference_conv2d_4_layer_call_fn_221802

inputs#
unknown:ЦЦ
	unknown_0:	Ц
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_2202392
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
¶
В
D__inference_conv2d_7_layer_call_and_return_conditional_losses_220332

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-220325*L
_output_shapes:
8:€€€€€€€€€<<Ц:€€€€€€€€€<<Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€<<Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€<<Ц
 
_user_specified_nameinputs
„
L
0__inference_up_sampling2d_1_layer_call_fn_221882

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_2199472
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ђ
g
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_221987

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"<   <   2
Constc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2	
Const_1X
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:2
mulљ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ*
half_pixel_centers(2
resize/ResizeNearestNeighborМ
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€<<Ц:X T
0
_output_shapes
:€€€€€€€€€<<Ц
 
_user_specified_nameinputs
я
Е
&__inference_model_layer_call_fn_221518
inputs_0
inputs_1"
unknown:Ц
	unknown_0:	Ц%
	unknown_1:ЦЦ
	unknown_2:	Ц%
	unknown_3:ЦЦ
	unknown_4:	Ц%
	unknown_5:ЦЦ
	unknown_6:	Ц
	unknown_7:
ЏЎ
	unknown_8:	Ў%
	unknown_9:ЦЦ

unknown_10:	Ц&

unknown_11:ЦЦ

unknown_12:	Ц&

unknown_13:ЦЦ

unknown_14:	Ц&

unknown_15:ЦЦ

unknown_16:	Ц%

unknown_17:Ц

unknown_18:
identity

identity_1ИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::€€€€€€€€€ђђ:€€€€€€€€€ђђ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2207812
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

IdentityЙ

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:€€€€€€€€€ђђ:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:€€€€€€€€€ђђ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
•
Я
)__inference_conv2d_8_layer_call_fn_222016

inputs"
unknown:Ц
	unknown_0:
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2203592
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":€€€€€€€€€ђђЦ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:€€€€€€€€€ђђЦ
 
_user_specified_nameinputs
эk
ƒ	
A__inference_model_layer_call_and_return_conditional_losses_220948
input_1
input_2(
conv2d_220878:Ц
conv2d_220880:	Ц+
conv2d_1_220884:ЦЦ
conv2d_1_220886:	Ц+
conv2d_2_220890:ЦЦ
conv2d_2_220892:	Ц+
conv2d_3_220896:ЦЦ
conv2d_3_220898:	Ц 
dense_220905:
ЏЎ
dense_220907:	Ў+
conv2d_4_220911:ЦЦ
conv2d_4_220913:	Ц+
conv2d_5_220917:ЦЦ
conv2d_5_220919:	Ц+
conv2d_6_220923:ЦЦ
conv2d_6_220925:	Ц+
conv2d_7_220929:ЦЦ
conv2d_7_220931:	Ц*
conv2d_8_220937:Ц
conv2d_8_220939:
identity

identity_1ИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallҐdense/StatefulPartitionedCallЦ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_220878conv2d_220880*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:€€€€€€€€€ђђЦ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_2200642 
conv2d/StatefulPartitionedCallН
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_2200742
max_pooling2d/PartitionedCallљ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_220884conv2d_1_220886*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2200922"
 conv2d_1/StatefulPartitionedCallХ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2201022!
max_pooling2d_1/PartitionedCallњ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_220890conv2d_2_220892*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2201202"
 conv2d_2/StatefulPartitionedCallХ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2201302!
max_pooling2d_2/PartitionedCallњ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_220896conv2d_3_220898*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2201482"
 conv2d_3/StatefulPartitionedCallХ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2201582!
max_pooling2d_3/PartitionedCallЎ
flatten_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2201662
flatten_1/PartitionedCallф
flatten/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ў* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2201742
flatten/PartitionedCallЭ
concatenate/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Џ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2201832
concatenate/PartitionedCall§
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_220905dense_220907*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ў*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2202012
dense/StatefulPartitionedCallъ
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_2202212
reshape/PartitionedCallЈ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_4_220911conv2d_4_220913*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_2202392"
 conv2d_4/StatefulPartitionedCallП
up_sampling2d/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_2202522
up_sampling2d/PartitionedCallљ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_5_220917conv2d_5_220919*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_2202702"
 conv2d_5/StatefulPartitionedCallХ
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_2202832!
up_sampling2d_1/PartitionedCallњ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_6_220923conv2d_6_220925*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2203012"
 conv2d_6/StatefulPartitionedCallХ
up_sampling2d_2/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2203142!
up_sampling2d_2/PartitionedCallњ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_7_220929conv2d_7_220931*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2203322"
 conv2d_7/StatefulPartitionedCall{
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
tf.math.greater/Greater/y¶
tf.math.greater/GreaterGreaterinput_1"tf.math.greater/Greater/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
tf.math.greater/GreaterЧ
up_sampling2d_3/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:€€€€€€€€€ђђЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2203472!
up_sampling2d_3/PartitionedCallј
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_8_220937conv2d_8_220939*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2203592"
 conv2d_8/StatefulPartitionedCallМ
tf.cast/CastCasttf.math.greater/Greater:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€ђђ2
tf.cast/Cast£
masking_layer/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0tf.cast/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_masking_layer_layer_call_and_return_conditional_losses_2203772
masking_layer/PartitionedCallю
lambda_1/PartitionedCallPartitionedCall&masking_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_2203872
lambda_1/PartitionedCallш
lambda/PartitionedCallPartitionedCall&masking_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2203972
lambda/PartitionedCallД
IdentityIdentitylambda/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

IdentityК

Identity_1Identity!lambda_1/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity_1І
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:€€€€€€€€€ђђ:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€ђђ
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_2
’
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_221704

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
 
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_222075

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2€
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
strided_slicet
IdentityIdentitystrided_slice:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ђђ:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
≤
g
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_221869

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ќ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul’
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(2
resize/ResizeNearestNeighbor§
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
 
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_220481

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2€
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
strided_slicet
IdentityIdentitystrided_slice:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ђђ:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
си
є
!__inference__wrapped_model_219807
input_1
input_2F
+model_conv2d_conv2d_readvariableop_resource:Ц;
,model_conv2d_biasadd_readvariableop_resource:	ЦI
-model_conv2d_1_conv2d_readvariableop_resource:ЦЦ=
.model_conv2d_1_biasadd_readvariableop_resource:	ЦI
-model_conv2d_2_conv2d_readvariableop_resource:ЦЦ=
.model_conv2d_2_biasadd_readvariableop_resource:	ЦI
-model_conv2d_3_conv2d_readvariableop_resource:ЦЦ=
.model_conv2d_3_biasadd_readvariableop_resource:	Ц>
*model_dense_matmul_readvariableop_resource:
ЏЎ:
+model_dense_biasadd_readvariableop_resource:	ЎI
-model_conv2d_4_conv2d_readvariableop_resource:ЦЦ=
.model_conv2d_4_biasadd_readvariableop_resource:	ЦI
-model_conv2d_5_conv2d_readvariableop_resource:ЦЦ=
.model_conv2d_5_biasadd_readvariableop_resource:	ЦI
-model_conv2d_6_conv2d_readvariableop_resource:ЦЦ=
.model_conv2d_6_biasadd_readvariableop_resource:	ЦI
-model_conv2d_7_conv2d_readvariableop_resource:ЦЦ=
.model_conv2d_7_biasadd_readvariableop_resource:	ЦH
-model_conv2d_8_conv2d_readvariableop_resource:Ц<
.model_conv2d_8_biasadd_readvariableop_resource:
identity

identity_1ИҐ#model/conv2d/BiasAdd/ReadVariableOpҐ"model/conv2d/Conv2D/ReadVariableOpҐ%model/conv2d_1/BiasAdd/ReadVariableOpҐ$model/conv2d_1/Conv2D/ReadVariableOpҐ%model/conv2d_2/BiasAdd/ReadVariableOpҐ$model/conv2d_2/Conv2D/ReadVariableOpҐ%model/conv2d_3/BiasAdd/ReadVariableOpҐ$model/conv2d_3/Conv2D/ReadVariableOpҐ%model/conv2d_4/BiasAdd/ReadVariableOpҐ$model/conv2d_4/Conv2D/ReadVariableOpҐ%model/conv2d_5/BiasAdd/ReadVariableOpҐ$model/conv2d_5/Conv2D/ReadVariableOpҐ%model/conv2d_6/BiasAdd/ReadVariableOpҐ$model/conv2d_6/Conv2D/ReadVariableOpҐ%model/conv2d_7/BiasAdd/ReadVariableOpҐ$model/conv2d_7/Conv2D/ReadVariableOpҐ%model/conv2d_8/BiasAdd/ReadVariableOpҐ$model/conv2d_8/Conv2D/ReadVariableOpҐ"model/dense/BiasAdd/ReadVariableOpҐ!model/dense/MatMul/ReadVariableOpљ
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:Ц*
dtype02$
"model/conv2d/Conv2D/ReadVariableOpќ
model/conv2d/Conv2DConv2Dinput_1*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ*
paddingSAME*
strides
2
model/conv2d/Conv2Dі
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOpњ
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2
model/conv2d/BiasAddУ
model/conv2d/SigmoidSigmoidmodel/conv2d/BiasAdd:output:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2
model/conv2d/Sigmoid°
model/conv2d/mulMulmodel/conv2d/BiasAdd:output:0model/conv2d/Sigmoid:y:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2
model/conv2d/mulН
model/conv2d/IdentityIdentitymodel/conv2d/mul:z:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2
model/conv2d/Identity€
model/conv2d/IdentityN	IdentityNmodel/conv2d/mul:z:0model/conv2d/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-219644*P
_output_shapes>
<:€€€€€€€€€ђђЦ:€€€€€€€€€ђђЦ2
model/conv2d/IdentityN‘
model/max_pooling2d/MaxPoolMaxPoolmodel/conv2d/IdentityN:output:0*0
_output_shapes
:€€€€€€€€€<<Ц*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d/MaxPoolƒ
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOpп
model/conv2d_1/Conv2DConv2D$model/max_pooling2d/MaxPool:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
paddingSAME*
strides
2
model/conv2d_1/Conv2DЇ
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp≈
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
model/conv2d_1/BiasAddЧ
model/conv2d_1/SigmoidSigmoidmodel/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
model/conv2d_1/SigmoidІ
model/conv2d_1/mulMulmodel/conv2d_1/BiasAdd:output:0model/conv2d_1/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
model/conv2d_1/mulС
model/conv2d_1/IdentityIdentitymodel/conv2d_1/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
model/conv2d_1/IdentityГ
model/conv2d_1/IdentityN	IdentityNmodel/conv2d_1/mul:z:0model/conv2d_1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-219657*L
_output_shapes:
8:€€€€€€€€€<<Ц:€€€€€€€€€<<Ц2
model/conv2d_1/IdentityNЏ
model/max_pooling2d_1/MaxPoolMaxPool!model/conv2d_1/IdentityN:output:0*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_1/MaxPoolƒ
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOpс
model/conv2d_2/Conv2DConv2D&model/max_pooling2d_1/MaxPool:output:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
model/conv2d_2/Conv2DЇ
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp≈
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_2/BiasAddЧ
model/conv2d_2/SigmoidSigmoidmodel/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_2/SigmoidІ
model/conv2d_2/mulMulmodel/conv2d_2/BiasAdd:output:0model/conv2d_2/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_2/mulС
model/conv2d_2/IdentityIdentitymodel/conv2d_2/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_2/IdentityГ
model/conv2d_2/IdentityN	IdentityNmodel/conv2d_2/mul:z:0model/conv2d_2/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-219670*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
model/conv2d_2/IdentityNЏ
model/max_pooling2d_2/MaxPoolMaxPool!model/conv2d_2/IdentityN:output:0*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_2/MaxPoolƒ
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02&
$model/conv2d_3/Conv2D/ReadVariableOpс
model/conv2d_3/Conv2DConv2D&model/max_pooling2d_2/MaxPool:output:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
model/conv2d_3/Conv2DЇ
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02'
%model/conv2d_3/BiasAdd/ReadVariableOp≈
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_3/BiasAddЧ
model/conv2d_3/SigmoidSigmoidmodel/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_3/SigmoidІ
model/conv2d_3/mulMulmodel/conv2d_3/BiasAdd:output:0model/conv2d_3/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_3/mulС
model/conv2d_3/IdentityIdentitymodel/conv2d_3/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_3/IdentityГ
model/conv2d_3/IdentityN	IdentityNmodel/conv2d_3/mul:z:0model/conv2d_3/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-219683*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
model/conv2d_3/IdentityNЏ
model/max_pooling2d_3/MaxPoolMaxPool!model/conv2d_3/IdentityN:output:0*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_3/MaxPool
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
model/flatten_1/ConstШ
model/flatten_1/ReshapeReshapeinput_2model/flatten_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
model/flatten_1/Reshape{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€X  2
model/flatten/Const≤
model/flatten/ReshapeReshape&model/max_pooling2d_3/MaxPool:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
model/flatten/ReshapeА
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axisж
model/concatenate/concatConcatV2 model/flatten_1/Reshape:output:0model/flatten/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Џ2
model/concatenate/concat≥
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
ЏЎ*
dtype02#
!model/dense/MatMul/ReadVariableOp≥
model/dense/MatMulMatMul!model/concatenate/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
model/dense/MatMul±
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ў*
dtype02$
"model/dense/BiasAdd/ReadVariableOp≤
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
model/dense/BiasAddЖ
model/dense/SigmoidSigmoidmodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
model/dense/SigmoidУ
model/dense/mulMulmodel/dense/BiasAdd:output:0model/dense/Sigmoid:y:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
model/dense/mulА
model/dense/IdentityIdentitymodel/dense/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
model/dense/Identityз
model/dense/IdentityN	IdentityNmodel/dense/mul:z:0model/dense/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-219702*<
_output_shapes*
(:€€€€€€€€€Ў:€€€€€€€€€Ў2
model/dense/IdentityNx
model/reshape/ShapeShapemodel/dense/IdentityN:output:0*
T0*
_output_shapes
:2
model/reshape/ShapeР
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model/reshape/strided_slice/stackФ
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_1Ф
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_2ґ
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/reshape/strided_sliceА
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
model/reshape/Reshape/shape/1А
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
model/reshape/Reshape/shape/2Б
model/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :Ц2
model/reshape/Reshape/shape/3О
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0&model/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
model/reshape/Reshape/shapeЇ
model/reshape/ReshapeReshapemodel/dense/IdentityN:output:0$model/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/reshape/Reshapeƒ
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02&
$model/conv2d_4/Conv2D/ReadVariableOpй
model/conv2d_4/Conv2DConv2Dmodel/reshape/Reshape:output:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
model/conv2d_4/Conv2DЇ
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02'
%model/conv2d_4/BiasAdd/ReadVariableOp≈
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_4/BiasAddЧ
model/conv2d_4/SigmoidSigmoidmodel/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_4/SigmoidІ
model/conv2d_4/mulMulmodel/conv2d_4/BiasAdd:output:0model/conv2d_4/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_4/mulС
model/conv2d_4/IdentityIdentitymodel/conv2d_4/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_4/IdentityГ
model/conv2d_4/IdentityN	IdentityNmodel/conv2d_4/mul:z:0model/conv2d_4/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-219724*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
model/conv2d_4/IdentityNЗ
model/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model/up_sampling2d/ConstЛ
model/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
model/up_sampling2d/Const_1®
model/up_sampling2d/mulMul"model/up_sampling2d/Const:output:0$model/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
model/up_sampling2d/mulТ
0model/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor!model/conv2d_4/IdentityN:output:0model/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
half_pixel_centers(22
0model/up_sampling2d/resize/ResizeNearestNeighborƒ
$model/conv2d_5/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02&
$model/conv2d_5/Conv2D/ReadVariableOpМ
model/conv2d_5/Conv2DConv2DAmodel/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0,model/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
model/conv2d_5/Conv2DЇ
%model/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02'
%model/conv2d_5/BiasAdd/ReadVariableOp≈
model/conv2d_5/BiasAddBiasAddmodel/conv2d_5/Conv2D:output:0-model/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_5/BiasAddЧ
model/conv2d_5/SigmoidSigmoidmodel/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_5/SigmoidІ
model/conv2d_5/mulMulmodel/conv2d_5/BiasAdd:output:0model/conv2d_5/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_5/mulС
model/conv2d_5/IdentityIdentitymodel/conv2d_5/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_5/IdentityГ
model/conv2d_5/IdentityN	IdentityNmodel/conv2d_5/mul:z:0model/conv2d_5/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-219740*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
model/conv2d_5/IdentityNЛ
model/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model/up_sampling2d_1/ConstП
model/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
model/up_sampling2d_1/Const_1∞
model/up_sampling2d_1/mulMul$model/up_sampling2d_1/Const:output:0&model/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
model/up_sampling2d_1/mulШ
2model/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor!model/conv2d_5/IdentityN:output:0model/up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
half_pixel_centers(24
2model/up_sampling2d_1/resize/ResizeNearestNeighborƒ
$model/conv2d_6/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02&
$model/conv2d_6/Conv2D/ReadVariableOpО
model/conv2d_6/Conv2DConv2DCmodel/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0,model/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
model/conv2d_6/Conv2DЇ
%model/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02'
%model/conv2d_6/BiasAdd/ReadVariableOp≈
model/conv2d_6/BiasAddBiasAddmodel/conv2d_6/Conv2D:output:0-model/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_6/BiasAddЧ
model/conv2d_6/SigmoidSigmoidmodel/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_6/SigmoidІ
model/conv2d_6/mulMulmodel/conv2d_6/BiasAdd:output:0model/conv2d_6/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_6/mulС
model/conv2d_6/IdentityIdentitymodel/conv2d_6/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
model/conv2d_6/IdentityГ
model/conv2d_6/IdentityN	IdentityNmodel/conv2d_6/mul:z:0model/conv2d_6/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-219756*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
model/conv2d_6/IdentityNЛ
model/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model/up_sampling2d_2/ConstП
model/up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
model/up_sampling2d_2/Const_1∞
model/up_sampling2d_2/mulMul$model/up_sampling2d_2/Const:output:0&model/up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:2
model/up_sampling2d_2/mulШ
2model/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor!model/conv2d_6/IdentityN:output:0model/up_sampling2d_2/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
half_pixel_centers(24
2model/up_sampling2d_2/resize/ResizeNearestNeighborƒ
$model/conv2d_7/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02&
$model/conv2d_7/Conv2D/ReadVariableOpО
model/conv2d_7/Conv2DConv2DCmodel/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0,model/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
paddingSAME*
strides
2
model/conv2d_7/Conv2DЇ
%model/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02'
%model/conv2d_7/BiasAdd/ReadVariableOp≈
model/conv2d_7/BiasAddBiasAddmodel/conv2d_7/Conv2D:output:0-model/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
model/conv2d_7/BiasAddЧ
model/conv2d_7/SigmoidSigmoidmodel/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
model/conv2d_7/SigmoidІ
model/conv2d_7/mulMulmodel/conv2d_7/BiasAdd:output:0model/conv2d_7/Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
model/conv2d_7/mulС
model/conv2d_7/IdentityIdentitymodel/conv2d_7/mul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
model/conv2d_7/IdentityГ
model/conv2d_7/IdentityN	IdentityNmodel/conv2d_7/mul:z:0model/conv2d_7/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-219772*L
_output_shapes:
8:€€€€€€€€€<<Ц:€€€€€€€€€<<Ц2
model/conv2d_7/IdentityNЗ
model/tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
model/tf.math.greater/Greater/yЄ
model/tf.math.greater/GreaterGreaterinput_1(model/tf.math.greater/Greater/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
model/tf.math.greater/GreaterЛ
model/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"<   <   2
model/up_sampling2d_3/ConstП
model/up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
model/up_sampling2d_3/Const_1∞
model/up_sampling2d_3/mulMul$model/up_sampling2d_3/Const:output:0&model/up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:2
model/up_sampling2d_3/mulЪ
2model/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor!model/conv2d_7/IdentityN:output:0model/up_sampling2d_3/mul:z:0*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ*
half_pixel_centers(24
2model/up_sampling2d_3/resize/ResizeNearestNeighbor√
$model/conv2d_8/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_8_conv2d_readvariableop_resource*'
_output_shapes
:Ц*
dtype02&
$model/conv2d_8/Conv2D/ReadVariableOpП
model/conv2d_8/Conv2DConv2DCmodel/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0,model/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*
paddingSAME*
strides
2
model/conv2d_8/Conv2Dє
%model/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv2d_8/BiasAdd/ReadVariableOp∆
model/conv2d_8/BiasAddBiasAddmodel/conv2d_8/Conv2D:output:0-model/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
model/conv2d_8/BiasAddЮ
model/tf.cast/CastCast!model/tf.math.greater/Greater:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€ђђ2
model/tf.cast/Cast{
model/masking_layer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
model/masking_layer/sub/x±
model/masking_layer/subSub"model/masking_layer/sub/x:output:0model/tf.cast/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
model/masking_layer/subЃ
model/masking_layer/mulMulmodel/conv2d_8/BiasAdd:output:0model/tf.cast/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
model/masking_layer/mul
model/masking_layer/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/masking_layer/mul_1/yЉ
model/masking_layer/mul_1Mulmodel/masking_layer/sub:z:0$model/masking_layer/mul_1/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
model/masking_layer/mul_1≥
model/masking_layer/addAddV2model/masking_layer/mul:z:0model/masking_layer/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
model/masking_layer/add°
"model/lambda_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2$
"model/lambda_1/strided_slice/stack•
$model/lambda_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2&
$model/lambda_1/strided_slice/stack_1•
$model/lambda_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2&
$model/lambda_1/strided_slice/stack_2я
model/lambda_1/strided_sliceStridedSlicemodel/masking_layer/add:z:0+model/lambda_1/strided_slice/stack:output:0-model/lambda_1/strided_slice/stack_1:output:0-model/lambda_1/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
model/lambda_1/strided_sliceЭ
 model/lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2"
 model/lambda/strided_slice/stack°
"model/lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2$
"model/lambda/strided_slice/stack_1°
"model/lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2$
"model/lambda/strided_slice/stack_2’
model/lambda/strided_sliceStridedSlicemodel/masking_layer/add:z:0)model/lambda/strided_slice/stack:output:0+model/lambda/strided_slice/stack_1:output:0+model/lambda/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
model/lambda/strided_sliceИ
IdentityIdentity#model/lambda/strided_slice:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

IdentityО

Identity_1Identity%model/lambda_1/strided_slice:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity_1Џ
NoOpNoOp$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp&^model/conv2d_5/BiasAdd/ReadVariableOp%^model/conv2d_5/Conv2D/ReadVariableOp&^model/conv2d_6/BiasAdd/ReadVariableOp%^model/conv2d_6/Conv2D/ReadVariableOp&^model/conv2d_7/BiasAdd/ReadVariableOp%^model/conv2d_7/Conv2D/ReadVariableOp&^model/conv2d_8/BiasAdd/ReadVariableOp%^model/conv2d_8/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:€€€€€€€€€ђђ:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp2N
%model/conv2d_5/BiasAdd/ReadVariableOp%model/conv2d_5/BiasAdd/ReadVariableOp2L
$model/conv2d_5/Conv2D/ReadVariableOp$model/conv2d_5/Conv2D/ReadVariableOp2N
%model/conv2d_6/BiasAdd/ReadVariableOp%model/conv2d_6/BiasAdd/ReadVariableOp2L
$model/conv2d_6/Conv2D/ReadVariableOp$model/conv2d_6/Conv2D/ReadVariableOp2N
%model/conv2d_7/BiasAdd/ReadVariableOp%model/conv2d_7/BiasAdd/ReadVariableOp2L
$model/conv2d_7/Conv2D/ReadVariableOp$model/conv2d_7/Conv2D/ReadVariableOp2N
%model/conv2d_8/BiasAdd/ReadVariableOp%model/conv2d_8/BiasAdd/ReadVariableOp2L
$model/conv2d_8/Conv2D/ReadVariableOp$model/conv2d_8/Conv2D/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp:Z V
1
_output_shapes
:€€€€€€€€€ђђ
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_2
≤
g
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_220019

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ќ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul’
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(2
resize/ResizeNearestNeighbor§
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
¶
В
D__inference_conv2d_4_layer_call_and_return_conditional_losses_220239

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-220232*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
∞
e
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_221814

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ќ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul’
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(2
resize/ResizeNearestNeighbor§
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ј
Б
$__inference_signature_wrapper_221076
input_1
input_2"
unknown:Ц
	unknown_0:	Ц%
	unknown_1:ЦЦ
	unknown_2:	Ц%
	unknown_3:ЦЦ
	unknown_4:	Ц%
	unknown_5:ЦЦ
	unknown_6:	Ц
	unknown_7:
ЏЎ
	unknown_8:	Ў%
	unknown_9:ЦЦ

unknown_10:	Ц&

unknown_11:ЦЦ

unknown_12:	Ц&

unknown_13:ЦЦ

unknown_14:	Ц&

unknown_15:ЦЦ

unknown_16:	Ц%

unknown_17:Ц

unknown_18:
identity

identity_1ИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::€€€€€€€€€ђђ:€€€€€€€€€ђђ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_2198072
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

IdentityЙ

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:€€€€€€€€€ђђ:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€ђђ
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_2
¶
В
D__inference_conv2d_6_layer_call_and_return_conditional_losses_220301

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-220294*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
Ы
ч
A__inference_dense_layer_call_and_return_conditional_losses_221749

inputs2
matmul_readvariableop_resource:
ЏЎ.
biasadd_readvariableop_resource:	Ў

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЏЎ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ў*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2	
Sigmoidc
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2

IdentityЈ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221742*<
_output_shapes*
(:€€€€€€€€€Ў:€€€€€€€€€Ў2
	IdentityNr

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Ў2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Џ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Џ
 
_user_specified_nameinputs
Ф
ю
D__inference_conv2d_8_layer_call_and_return_conditional_losses_220359

inputs9
conv2d_readvariableop_resource:Ц-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:Ц*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":€€€€€€€€€ђђЦ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:€€€€€€€€€ђђЦ
 
_user_specified_nameinputs
Ґ
Ю
'__inference_conv2d_layer_call_fn_221543

inputs"
unknown:Ц
	unknown_0:	Ц
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:€€€€€€€€€ђђЦ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_2200642
StatefulPartitionedCallЖ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:€€€€€€€€€ђђЦ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€ђђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
¶
В
D__inference_conv2d_1_layer_call_and_return_conditional_losses_220092

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-220085*L
_output_shapes:
8:€€€€€€€€€<<Ц:€€€€€€€€€<<Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€<<Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€<<Ц
 
_user_specified_nameinputs
эk
ƒ	
A__inference_model_layer_call_and_return_conditional_losses_221022
input_1
input_2(
conv2d_220952:Ц
conv2d_220954:	Ц+
conv2d_1_220958:ЦЦ
conv2d_1_220960:	Ц+
conv2d_2_220964:ЦЦ
conv2d_2_220966:	Ц+
conv2d_3_220970:ЦЦ
conv2d_3_220972:	Ц 
dense_220979:
ЏЎ
dense_220981:	Ў+
conv2d_4_220985:ЦЦ
conv2d_4_220987:	Ц+
conv2d_5_220991:ЦЦ
conv2d_5_220993:	Ц+
conv2d_6_220997:ЦЦ
conv2d_6_220999:	Ц+
conv2d_7_221003:ЦЦ
conv2d_7_221005:	Ц*
conv2d_8_221011:Ц
conv2d_8_221013:
identity

identity_1ИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallҐdense/StatefulPartitionedCallЦ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_220952conv2d_220954*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:€€€€€€€€€ђђЦ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_2200642 
conv2d/StatefulPartitionedCallН
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_2200742
max_pooling2d/PartitionedCallљ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_220958conv2d_1_220960*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2200922"
 conv2d_1/StatefulPartitionedCallХ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2201022!
max_pooling2d_1/PartitionedCallњ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_220964conv2d_2_220966*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2201202"
 conv2d_2/StatefulPartitionedCallХ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2201302!
max_pooling2d_2/PartitionedCallњ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_220970conv2d_3_220972*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2201482"
 conv2d_3/StatefulPartitionedCallХ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2201582!
max_pooling2d_3/PartitionedCallЎ
flatten_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2201662
flatten_1/PartitionedCallф
flatten/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ў* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2201742
flatten/PartitionedCallЭ
concatenate/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Џ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2201832
concatenate/PartitionedCall§
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_220979dense_220981*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ў*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2202012
dense/StatefulPartitionedCallъ
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_2202212
reshape/PartitionedCallЈ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_4_220985conv2d_4_220987*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_2202392"
 conv2d_4/StatefulPartitionedCallП
up_sampling2d/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_2202522
up_sampling2d/PartitionedCallљ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_5_220991conv2d_5_220993*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_2202702"
 conv2d_5/StatefulPartitionedCallХ
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_2202832!
up_sampling2d_1/PartitionedCallњ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_6_220997conv2d_6_220999*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2203012"
 conv2d_6/StatefulPartitionedCallХ
up_sampling2d_2/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2203142!
up_sampling2d_2/PartitionedCallњ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_7_221003conv2d_7_221005*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2203322"
 conv2d_7/StatefulPartitionedCall{
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
tf.math.greater/Greater/y¶
tf.math.greater/GreaterGreaterinput_1"tf.math.greater/Greater/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
tf.math.greater/GreaterЧ
up_sampling2d_3/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:€€€€€€€€€ђђЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2203472!
up_sampling2d_3/PartitionedCallј
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_8_221011conv2d_8_221013*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2203592"
 conv2d_8/StatefulPartitionedCallМ
tf.cast/CastCasttf.math.greater/Greater:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€ђђ2
tf.cast/Cast£
masking_layer/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0tf.cast/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_masking_layer_layer_call_and_return_conditional_losses_2203772
masking_layer/PartitionedCallю
lambda_1/PartitionedCallPartitionedCall&masking_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_2204812
lambda_1/PartitionedCallш
lambda/PartitionedCallPartitionedCall&masking_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ђђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2204622
lambda/PartitionedCallД
IdentityIdentitylambda/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

IdentityК

Identity_1Identity!lambda_1/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity_1І
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:€€€€€€€€€ђђ:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€ђђ
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_2
¶
В
D__inference_conv2d_2_layer_call_and_return_conditional_losses_220120

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-220113*L
_output_shapes:
8:€€€€€€€€€Ц:€€€€€€€€€Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
¶
e
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_221822

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Constc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2	
Const_1X
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:2
mulї
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
half_pixel_centers(2
resize/ResizeNearestNeighborК
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
шМ
™
"__inference__traced_restore_222324
file_prefix9
assignvariableop_conv2d_kernel:Ц-
assignvariableop_1_conv2d_bias:	Ц>
"assignvariableop_2_conv2d_1_kernel:ЦЦ/
 assignvariableop_3_conv2d_1_bias:	Ц>
"assignvariableop_4_conv2d_2_kernel:ЦЦ/
 assignvariableop_5_conv2d_2_bias:	Ц>
"assignvariableop_6_conv2d_3_kernel:ЦЦ/
 assignvariableop_7_conv2d_3_bias:	Ц3
assignvariableop_8_dense_kernel:
ЏЎ,
assignvariableop_9_dense_bias:	Ў?
#assignvariableop_10_conv2d_4_kernel:ЦЦ0
!assignvariableop_11_conv2d_4_bias:	Ц?
#assignvariableop_12_conv2d_5_kernel:ЦЦ0
!assignvariableop_13_conv2d_5_bias:	Ц?
#assignvariableop_14_conv2d_6_kernel:ЦЦ0
!assignvariableop_15_conv2d_6_bias:	Ц?
#assignvariableop_16_conv2d_7_kernel:ЦЦ0
!assignvariableop_17_conv2d_7_bias:	Ц>
#assignvariableop_18_conv2d_8_kernel:Ц/
!assignvariableop_19_conv2d_8_bias:#
assignvariableop_20_decay: +
!assignvariableop_21_learning_rate: &
assignvariableop_22_momentum: &
assignvariableop_23_sgd_iter:	 #
assignvariableop_24_total: #
assignvariableop_25_count: %
assignvariableop_26_total_1: %
assignvariableop_27_count_1: %
assignvariableop_28_total_2: %
assignvariableop_29_count_2: %
assignvariableop_30_total_3: %
assignvariableop_31_count_3: %
assignvariableop_32_total_4: %
assignvariableop_33_count_4: 
identity_35ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9…
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*’
valueЋB»#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names‘
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЁ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ґ
_output_shapesП
М:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЭ
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2І
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3•
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4І
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5•
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6І
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7•
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ґ
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ђ
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11©
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ђ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_5_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_5_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ђ
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_6_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_6_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ђ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_7_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_7_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ђ
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_8_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19©
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_8_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20°
AssignVariableOp_20AssignVariableOpassignvariableop_20_decayIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21©
AssignVariableOp_21AssignVariableOp!assignvariableop_21_learning_rateIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22§
AssignVariableOp_22AssignVariableOpassignvariableop_22_momentumIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_23§
AssignVariableOp_23AssignVariableOpassignvariableop_23_sgd_iterIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24°
AssignVariableOp_24AssignVariableOpassignvariableop_24_totalIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25°
AssignVariableOp_25AssignVariableOpassignvariableop_25_countIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26£
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27£
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28£
AssignVariableOp_28AssignVariableOpassignvariableop_28_total_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29£
AssignVariableOp_29AssignVariableOpassignvariableop_29_count_2Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30£
AssignVariableOp_30AssignVariableOpassignvariableop_30_total_3Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31£
AssignVariableOp_31AssignVariableOpassignvariableop_31_count_3Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32£
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_4Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33£
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_4Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_339
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp 
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34f
Identity_35IdentityIdentity_34:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_35≤
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_35Identity_35:output:0*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Њ
F
*__inference_flatten_1_layer_call_fn_221709

inputs
identity√
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2201662
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ќ
D
(__inference_flatten_layer_call_fn_221720

inputs
identity¬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ў* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2201742
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
Ф
ю
D__inference_conv2d_8_layer_call_and_return_conditional_losses_222007

inputs9
conv2d_readvariableop_resource:Ц-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:Ц*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":€€€€€€€€€ђђЦ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:€€€€€€€€€ђђЦ
 
_user_specified_nameinputs
ў
Г
&__inference_model_layer_call_fn_220446
input_1
input_2"
unknown:Ц
	unknown_0:	Ц%
	unknown_1:ЦЦ
	unknown_2:	Ц%
	unknown_3:ЦЦ
	unknown_4:	Ц%
	unknown_5:ЦЦ
	unknown_6:	Ц
	unknown_7:
ЏЎ
	unknown_8:	Ў%
	unknown_9:ЦЦ

unknown_10:	Ц&

unknown_11:ЦЦ

unknown_12:	Ц&

unknown_13:ЦЦ

unknown_14:	Ц&

unknown_15:ЦЦ

unknown_16:	Ц%

unknown_17:Ц

unknown_18:
identity

identity_1ИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::€€€€€€€€€ђђ:€€€€€€€€€ђђ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2204012
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

IdentityЙ

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:€€€€€€€€€ђђ:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€ђђ
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_2
ƒ
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_221553

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€<<Ц*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ђђЦ:Z V
2
_output_shapes 
:€€€€€€€€€ђђЦ
 
_user_specified_nameinputs
¶
В
D__inference_conv2d_7_layer_call_and_return_conditional_losses_221958

inputs:
conv2d_readvariableop_resource:ЦЦ.
biasadd_readvariableop_resource:	Ц

identity_1ИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ЦЦ*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2	
BiasAddj
SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2	
Sigmoidk
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2
muld
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity«
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-221951*L
_output_shapes:
8:€€€€€€€€€<<Ц:€€€€€€€€€<<Ц2
	IdentityNz

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€<<Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€<<Ц
 
_user_specified_nameinputs
 
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_220387

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2€
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ђђ*

begin_mask*
end_mask2
strided_slicet
IdentityIdentitystrided_slice:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ђђ:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
®
g
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_221877

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Constc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2	
Const_1X
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:2
mulї
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*0
_output_shapes
:€€€€€€€€€Ц*
half_pixel_centers(2
resize/ResizeNearestNeighborК
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
т
Ц
&__inference_dense_layer_call_fn_221758

inputs
unknown:
ЏЎ
	unknown_0:	Ў
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ў*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2202012
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Ў2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Џ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Џ
 
_user_specified_nameinputs
х
s
I__inference_masking_layer_layer_call_and_return_conditional_losses_220377

inputs
inputs_1
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xg
subSubsub/x:output:0inputs_1*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
sub_
mulMulinputsinputs_1*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
mulW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
mul_1/yl
mul_1Mulsub:z:0mul_1/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
mul_1c
addAddV2mul:z:0	mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:€€€€€€€€€ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::€€€€€€€€€ђђ:€€€€€€€€€ђђ:Y U
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs:YU
1
_output_shapes
:€€€€€€€€€ђђ
 
_user_specified_nameinputs
Ђ
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_221593

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≤
g
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_219983

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ќ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul’
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(2
resize/ResizeNearestNeighbor§
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
¬
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_220158

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
¬
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_221598

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€<<Ц:X T
0
_output_shapes
:€€€€€€€€€<<Ц
 
_user_specified_nameinputs
°
°
)__inference_conv2d_3_layer_call_fn_221678

inputs#
unknown:ЦЦ
	unknown_0:	Ц
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2201482
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€Ц: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
≤
g
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_221979

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ќ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul’
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(2
resize/ResizeNearestNeighbor§
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
о
J
.__inference_max_pooling2d_layer_call_fn_221563

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€<<Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_2200742
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ђђЦ:Z V
2
_output_shapes 
:€€€€€€€€€ђђЦ
 
_user_specified_nameinputs
к
s
G__inference_concatenate_layer_call_and_return_conditional_losses_221727
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisВ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Џ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Џ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€:€€€€€€€€€Ў:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:€€€€€€€€€Ў
"
_user_specified_name
inputs/1
©
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_219816

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
”
J
.__inference_up_sampling2d_layer_call_fn_221827

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_2199112
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
о
L
0__inference_up_sampling2d_1_layer_call_fn_221887

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_2202832
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
¬
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_220130

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
„
L
0__inference_max_pooling2d_3_layer_call_fn_221693

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2198822
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ƒ
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_220074

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€<<Ц*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€ђђЦ:Z V
2
_output_shapes 
:€€€€€€€€€ђђЦ
 
_user_specified_nameinputs
¬
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_221688

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€Ц*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Ц:X T
0
_output_shapes
:€€€€€€€€€Ц
 
_user_specified_nameinputs
Ы
ч
A__inference_dense_layer_call_and_return_conditional_losses_220201

inputs2
matmul_readvariableop_resource:
ЏЎ.
biasadd_readvariableop_resource:	Ў

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЏЎ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ў*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2	
Sigmoidc
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:€€€€€€€€€Ў2

IdentityЈ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-220194*<
_output_shapes*
(:€€€€€€€€€Ў:€€€€€€€€€Ў2
	IdentityNr

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Ў2

Identity_1
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Џ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Џ
 
_user_specified_nameinputs"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_defaultЃ
E
input_1:
serving_default_input_1:0€€€€€€€€€ђђ
;
input_20
serving_default_input_2:0€€€€€€€€€D
lambda:
StatefulPartitionedCall:0€€€€€€€€€ђђF
lambda_1:
StatefulPartitionedCall:1€€€€€€€€€ђђtensorflow/serving/predict:т†
ц
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer_with_weights-6
layer-17
layer-18
layer_with_weights-7
layer-19
layer-20
layer_with_weights-8
layer-21
layer-22
layer-23
layer_with_weights-9
layer-24
layer-25
layer-26
layer-27
layer-28
	optimizer
regularization_losses
 trainable_variables
!	variables
"	keras_api
#
signatures
њ_default_save_signature
+ј&call_and_return_all_conditional_losses
Ѕ__call__"
_tf_keras_network
"
_tf_keras_input_layer
љ

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
+¬&call_and_return_all_conditional_losses
√__call__"
_tf_keras_layer
І
*regularization_losses
+trainable_variables
,	variables
-	keras_api
+ƒ&call_and_return_all_conditional_losses
≈__call__"
_tf_keras_layer
љ

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
+∆&call_and_return_all_conditional_losses
«__call__"
_tf_keras_layer
І
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+»&call_and_return_all_conditional_losses
…__call__"
_tf_keras_layer
љ

8kernel
9bias
:regularization_losses
;trainable_variables
<	variables
=	keras_api
+ &call_and_return_all_conditional_losses
Ћ__call__"
_tf_keras_layer
І
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+ћ&call_and_return_all_conditional_losses
Ќ__call__"
_tf_keras_layer
љ

Bkernel
Cbias
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
+ќ&call_and_return_all_conditional_losses
ѕ__call__"
_tf_keras_layer
"
_tf_keras_input_layer
І
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
+–&call_and_return_all_conditional_losses
—__call__"
_tf_keras_layer
І
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
+“&call_and_return_all_conditional_losses
”__call__"
_tf_keras_layer
І
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
+‘&call_and_return_all_conditional_losses
’__call__"
_tf_keras_layer
І
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
+÷&call_and_return_all_conditional_losses
„__call__"
_tf_keras_layer
љ

Xkernel
Ybias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
+Ў&call_and_return_all_conditional_losses
ў__call__"
_tf_keras_layer
І
^regularization_losses
_trainable_variables
`	variables
a	keras_api
+Џ&call_and_return_all_conditional_losses
џ__call__"
_tf_keras_layer
љ

bkernel
cbias
dregularization_losses
etrainable_variables
f	variables
g	keras_api
+№&call_and_return_all_conditional_losses
Ё__call__"
_tf_keras_layer
І
hregularization_losses
itrainable_variables
j	variables
k	keras_api
+ё&call_and_return_all_conditional_losses
я__call__"
_tf_keras_layer
љ

lkernel
mbias
nregularization_losses
otrainable_variables
p	variables
q	keras_api
+а&call_and_return_all_conditional_losses
б__call__"
_tf_keras_layer
І
rregularization_losses
strainable_variables
t	variables
u	keras_api
+в&call_and_return_all_conditional_losses
г__call__"
_tf_keras_layer
љ

vkernel
wbias
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
+д&call_and_return_all_conditional_losses
е__call__"
_tf_keras_layer
І
|regularization_losses
}trainable_variables
~	variables
	keras_api
+ж&call_and_return_all_conditional_losses
з__call__"
_tf_keras_layer
√
Аkernel
	Бbias
Вregularization_losses
Гtrainable_variables
Д	variables
Е	keras_api
+и&call_and_return_all_conditional_losses
й__call__"
_tf_keras_layer
Ђ
Жregularization_losses
Зtrainable_variables
И	variables
Й	keras_api
+к&call_and_return_all_conditional_losses
л__call__"
_tf_keras_layer
)
К	keras_api"
_tf_keras_layer
√
Лkernel
	Мbias
Нregularization_losses
Оtrainable_variables
П	variables
Р	keras_api
+м&call_and_return_all_conditional_losses
н__call__"
_tf_keras_layer
)
С	keras_api"
_tf_keras_layer
Ђ
Тregularization_losses
Уtrainable_variables
Ф	variables
Х	keras_api
+о&call_and_return_all_conditional_losses
п__call__"
_tf_keras_layer
Ђ
Цregularization_losses
Чtrainable_variables
Ш	variables
Щ	keras_api
+р&call_and_return_all_conditional_losses
с__call__"
_tf_keras_layer
Ђ
Ъregularization_losses
Ыtrainable_variables
Ь	variables
Э	keras_api
+т&call_and_return_all_conditional_losses
у__call__"
_tf_keras_layer
M

Юdecay
Яlearning_rate
†momentum
	°iter"
	optimizer
 "
trackable_list_wrapper
Ї
$0
%1
.2
/3
84
95
B6
C7
X8
Y9
b10
c11
l12
m13
v14
w15
А16
Б17
Л18
М19"
trackable_list_wrapper
Ї
$0
%1
.2
/3
84
95
B6
C7
X8
Y9
b10
c11
l12
m13
v14
w15
А16
Б17
Л18
М19"
trackable_list_wrapper
”
Ґmetrics
regularization_losses
 trainable_variables
!	variables
£non_trainable_variables
 §layer_regularization_losses
•layer_metrics
¶layers
Ѕ__call__
њ_default_save_signature
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
-
фserving_default"
signature_map
(:&Ц2conv2d/kernel
:Ц2conv2d/bias
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
µ
Іmetrics
&regularization_losses
'trainable_variables
(	variables
®non_trainable_variables
 ©layer_regularization_losses
™layer_metrics
Ђlayers
√__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ђmetrics
*regularization_losses
+trainable_variables
,	variables
≠non_trainable_variables
 Ѓlayer_regularization_losses
ѓlayer_metrics
∞layers
≈__call__
+ƒ&call_and_return_all_conditional_losses
'ƒ"call_and_return_conditional_losses"
_generic_user_object
+:)ЦЦ2conv2d_1/kernel
:Ц2conv2d_1/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
µ
±metrics
0regularization_losses
1trainable_variables
2	variables
≤non_trainable_variables
 ≥layer_regularization_losses
іlayer_metrics
µlayers
«__call__
+∆&call_and_return_all_conditional_losses
'∆"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ґmetrics
4regularization_losses
5trainable_variables
6	variables
Јnon_trainable_variables
 Єlayer_regularization_losses
єlayer_metrics
Їlayers
…__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
+:)ЦЦ2conv2d_2/kernel
:Ц2conv2d_2/bias
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
µ
їmetrics
:regularization_losses
;trainable_variables
<	variables
Љnon_trainable_variables
 љlayer_regularization_losses
Њlayer_metrics
њlayers
Ћ__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
јmetrics
>regularization_losses
?trainable_variables
@	variables
Ѕnon_trainable_variables
 ¬layer_regularization_losses
√layer_metrics
ƒlayers
Ќ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
+:)ЦЦ2conv2d_3/kernel
:Ц2conv2d_3/bias
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
µ
≈metrics
Dregularization_losses
Etrainable_variables
F	variables
∆non_trainable_variables
 «layer_regularization_losses
»layer_metrics
…layers
ѕ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 metrics
Hregularization_losses
Itrainable_variables
J	variables
Ћnon_trainable_variables
 ћlayer_regularization_losses
Ќlayer_metrics
ќlayers
—__call__
+–&call_and_return_all_conditional_losses
'–"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ѕmetrics
Lregularization_losses
Mtrainable_variables
N	variables
–non_trainable_variables
 —layer_regularization_losses
“layer_metrics
”layers
”__call__
+“&call_and_return_all_conditional_losses
'“"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
‘metrics
Pregularization_losses
Qtrainable_variables
R	variables
’non_trainable_variables
 ÷layer_regularization_losses
„layer_metrics
Ўlayers
’__call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ўmetrics
Tregularization_losses
Utrainable_variables
V	variables
Џnon_trainable_variables
 џlayer_regularization_losses
№layer_metrics
Ёlayers
„__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 :
ЏЎ2dense/kernel
:Ў2
dense/bias
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
µ
ёmetrics
Zregularization_losses
[trainable_variables
\	variables
яnon_trainable_variables
 аlayer_regularization_losses
бlayer_metrics
вlayers
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
гmetrics
^regularization_losses
_trainable_variables
`	variables
дnon_trainable_variables
 еlayer_regularization_losses
жlayer_metrics
зlayers
џ__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
+:)ЦЦ2conv2d_4/kernel
:Ц2conv2d_4/bias
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
µ
иmetrics
dregularization_losses
etrainable_variables
f	variables
йnon_trainable_variables
 кlayer_regularization_losses
лlayer_metrics
мlayers
Ё__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
нmetrics
hregularization_losses
itrainable_variables
j	variables
оnon_trainable_variables
 пlayer_regularization_losses
рlayer_metrics
сlayers
я__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
+:)ЦЦ2conv2d_5/kernel
:Ц2conv2d_5/bias
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
µ
тmetrics
nregularization_losses
otrainable_variables
p	variables
уnon_trainable_variables
 фlayer_regularization_losses
хlayer_metrics
цlayers
б__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
чmetrics
rregularization_losses
strainable_variables
t	variables
шnon_trainable_variables
 щlayer_regularization_losses
ъlayer_metrics
ыlayers
г__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
+:)ЦЦ2conv2d_6/kernel
:Ц2conv2d_6/bias
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
µ
ьmetrics
xregularization_losses
ytrainable_variables
z	variables
эnon_trainable_variables
 юlayer_regularization_losses
€layer_metrics
Аlayers
е__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Бmetrics
|regularization_losses
}trainable_variables
~	variables
Вnon_trainable_variables
 Гlayer_regularization_losses
Дlayer_metrics
Еlayers
з__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
+:)ЦЦ2conv2d_7/kernel
:Ц2conv2d_7/bias
 "
trackable_list_wrapper
0
А0
Б1"
trackable_list_wrapper
0
А0
Б1"
trackable_list_wrapper
Є
Жmetrics
Вregularization_losses
Гtrainable_variables
Д	variables
Зnon_trainable_variables
 Иlayer_regularization_losses
Йlayer_metrics
Кlayers
й__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Лmetrics
Жregularization_losses
Зtrainable_variables
И	variables
Мnon_trainable_variables
 Нlayer_regularization_losses
Оlayer_metrics
Пlayers
л__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
*:(Ц2conv2d_8/kernel
:2conv2d_8/bias
 "
trackable_list_wrapper
0
Л0
М1"
trackable_list_wrapper
0
Л0
М1"
trackable_list_wrapper
Є
Рmetrics
Нregularization_losses
Оtrainable_variables
П	variables
Сnon_trainable_variables
 Тlayer_regularization_losses
Уlayer_metrics
Фlayers
н__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Хmetrics
Тregularization_losses
Уtrainable_variables
Ф	variables
Цnon_trainable_variables
 Чlayer_regularization_losses
Шlayer_metrics
Щlayers
п__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ъmetrics
Цregularization_losses
Чtrainable_variables
Ш	variables
Ыnon_trainable_variables
 Ьlayer_regularization_losses
Эlayer_metrics
Юlayers
с__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Яmetrics
Ъregularization_losses
Ыtrainable_variables
Ь	variables
†non_trainable_variables
 °layer_regularization_losses
Ґlayer_metrics
£layers
у__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
H
§0
•1
¶2
І3
®4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ю
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
R

©total

™count
Ђ	variables
ђ	keras_api"
_tf_keras_metric
R

≠total

Ѓcount
ѓ	variables
∞	keras_api"
_tf_keras_metric
R

±total

≤count
≥	variables
і	keras_api"
_tf_keras_metric
c

µtotal

ґcount
Ј
_fn_kwargs
Є	variables
є	keras_api"
_tf_keras_metric
c

Їtotal

їcount
Љ
_fn_kwargs
љ	variables
Њ	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
©0
™1"
trackable_list_wrapper
.
Ђ	variables"
_generic_user_object
:  (2total
:  (2count
0
≠0
Ѓ1"
trackable_list_wrapper
.
ѓ	variables"
_generic_user_object
:  (2total
:  (2count
0
±0
≤1"
trackable_list_wrapper
.
≥	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
µ0
ґ1"
trackable_list_wrapper
.
Є	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ї0
ї1"
trackable_list_wrapper
.
љ	variables"
_generic_user_object
’B“
!__inference__wrapped_model_219807input_1input_2"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
A__inference_model_layer_call_and_return_conditional_losses_221249
A__inference_model_layer_call_and_return_conditional_losses_221422
A__inference_model_layer_call_and_return_conditional_losses_220948
A__inference_model_layer_call_and_return_conditional_losses_221022ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2г
&__inference_model_layer_call_fn_220446
&__inference_model_layer_call_fn_221470
&__inference_model_layer_call_fn_221518
&__inference_model_layer_call_fn_220874ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
м2й
B__inference_conv2d_layer_call_and_return_conditional_losses_221534Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_conv2d_layer_call_fn_221543Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Њ2ї
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_221548
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_221553Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
И2Е
.__inference_max_pooling2d_layer_call_fn_221558
.__inference_max_pooling2d_layer_call_fn_221563Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_1_layer_call_and_return_conditional_losses_221579Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_conv2d_1_layer_call_fn_221588Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
¬2њ
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_221593
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_221598Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
М2Й
0__inference_max_pooling2d_1_layer_call_fn_221603
0__inference_max_pooling2d_1_layer_call_fn_221608Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_2_layer_call_and_return_conditional_losses_221624Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_conv2d_2_layer_call_fn_221633Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
¬2њ
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_221638
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_221643Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
М2Й
0__inference_max_pooling2d_2_layer_call_fn_221648
0__inference_max_pooling2d_2_layer_call_fn_221653Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_3_layer_call_and_return_conditional_losses_221669Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_conv2d_3_layer_call_fn_221678Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
¬2њ
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_221683
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_221688Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
М2Й
0__inference_max_pooling2d_3_layer_call_fn_221693
0__inference_max_pooling2d_3_layer_call_fn_221698Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_flatten_1_layer_call_and_return_conditional_losses_221704Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
‘2—
*__inference_flatten_1_layer_call_fn_221709Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_flatten_layer_call_and_return_conditional_losses_221715Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_flatten_layer_call_fn_221720Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_concatenate_layer_call_and_return_conditional_losses_221727Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_concatenate_layer_call_fn_221733Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_dense_layer_call_and_return_conditional_losses_221749Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_dense_layer_call_fn_221758Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_reshape_layer_call_and_return_conditional_losses_221772Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_reshape_layer_call_fn_221777Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_4_layer_call_and_return_conditional_losses_221793Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_conv2d_4_layer_call_fn_221802Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Њ2ї
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_221814
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_221822Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
И2Е
.__inference_up_sampling2d_layer_call_fn_221827
.__inference_up_sampling2d_layer_call_fn_221832Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_5_layer_call_and_return_conditional_losses_221848Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_conv2d_5_layer_call_fn_221857Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
¬2њ
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_221869
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_221877Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
М2Й
0__inference_up_sampling2d_1_layer_call_fn_221882
0__inference_up_sampling2d_1_layer_call_fn_221887Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_6_layer_call_and_return_conditional_losses_221903Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_conv2d_6_layer_call_fn_221912Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
¬2њ
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_221924
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_221932Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
М2Й
0__inference_up_sampling2d_2_layer_call_fn_221937
0__inference_up_sampling2d_2_layer_call_fn_221942Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_7_layer_call_and_return_conditional_losses_221958Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_conv2d_7_layer_call_fn_221967Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
¬2њ
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_221979
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_221987Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
М2Й
0__inference_up_sampling2d_3_layer_call_fn_221992
0__inference_up_sampling2d_3_layer_call_fn_221997Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_8_layer_call_and_return_conditional_losses_222007Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_conv2d_8_layer_call_fn_222016Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_masking_layer_layer_call_and_return_conditional_losses_222027Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ў2’
.__inference_masking_layer_layer_call_fn_222033Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќ2Ћ
B__inference_lambda_layer_call_and_return_conditional_losses_222041
B__inference_lambda_layer_call_and_return_conditional_losses_222049ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ш2Х
'__inference_lambda_layer_call_fn_222054
'__inference_lambda_layer_call_fn_222059ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
D__inference_lambda_1_layer_call_and_return_conditional_losses_222067
D__inference_lambda_1_layer_call_and_return_conditional_losses_222075ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ь2Щ
)__inference_lambda_1_layer_call_fn_222080
)__inference_lambda_1_layer_call_fn_222085ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“Bѕ
$__inference_signature_wrapper_221076input_1input_2"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 Щ
!__inference__wrapped_model_219807у$%./89BCXYbclmvwАБЛМbҐ_
XҐU
SЪP
+К(
input_1€€€€€€€€€ђђ
!К
input_2€€€€€€€€€
™ "s™p
4
lambda*К'
lambda€€€€€€€€€ђђ
8
lambda_1,К)
lambda_1€€€€€€€€€ђђ—
G__inference_concatenate_layer_call_and_return_conditional_losses_221727Е[ҐX
QҐN
LЪI
"К
inputs/0€€€€€€€€€
#К 
inputs/1€€€€€€€€€Ў
™ "&Ґ#
К
0€€€€€€€€€Џ
Ъ ®
,__inference_concatenate_layer_call_fn_221733x[ҐX
QҐN
LЪI
"К
inputs/0€€€€€€€€€
#К 
inputs/1€€€€€€€€€Ў
™ "К€€€€€€€€€Џґ
D__inference_conv2d_1_layer_call_and_return_conditional_losses_221579n./8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€<<Ц
™ ".Ґ+
$К!
0€€€€€€€€€<<Ц
Ъ О
)__inference_conv2d_1_layer_call_fn_221588a./8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€<<Ц
™ "!К€€€€€€€€€<<Цґ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_221624n898Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ ".Ґ+
$К!
0€€€€€€€€€Ц
Ъ О
)__inference_conv2d_2_layer_call_fn_221633a898Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ "!К€€€€€€€€€Цґ
D__inference_conv2d_3_layer_call_and_return_conditional_losses_221669nBC8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ ".Ґ+
$К!
0€€€€€€€€€Ц
Ъ О
)__inference_conv2d_3_layer_call_fn_221678aBC8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ "!К€€€€€€€€€Цґ
D__inference_conv2d_4_layer_call_and_return_conditional_losses_221793nbc8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ ".Ґ+
$К!
0€€€€€€€€€Ц
Ъ О
)__inference_conv2d_4_layer_call_fn_221802abc8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ "!К€€€€€€€€€Цґ
D__inference_conv2d_5_layer_call_and_return_conditional_losses_221848nlm8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ ".Ґ+
$К!
0€€€€€€€€€Ц
Ъ О
)__inference_conv2d_5_layer_call_fn_221857alm8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ "!К€€€€€€€€€Цґ
D__inference_conv2d_6_layer_call_and_return_conditional_losses_221903nvw8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ ".Ґ+
$К!
0€€€€€€€€€Ц
Ъ О
)__inference_conv2d_6_layer_call_fn_221912avw8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ "!К€€€€€€€€€ЦЄ
D__inference_conv2d_7_layer_call_and_return_conditional_losses_221958pАБ8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€<<Ц
™ ".Ґ+
$К!
0€€€€€€€€€<<Ц
Ъ Р
)__inference_conv2d_7_layer_call_fn_221967cАБ8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€<<Ц
™ "!К€€€€€€€€€<<Цї
D__inference_conv2d_8_layer_call_and_return_conditional_losses_222007sЛМ:Ґ7
0Ґ-
+К(
inputs€€€€€€€€€ђђЦ
™ "/Ґ,
%К"
0€€€€€€€€€ђђ
Ъ У
)__inference_conv2d_8_layer_call_fn_222016fЛМ:Ґ7
0Ґ-
+К(
inputs€€€€€€€€€ђђЦ
™ ""К€€€€€€€€€ђђЈ
B__inference_conv2d_layer_call_and_return_conditional_losses_221534q$%9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€ђђ
™ "0Ґ-
&К#
0€€€€€€€€€ђђЦ
Ъ П
'__inference_conv2d_layer_call_fn_221543d$%9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€ђђ
™ "#К €€€€€€€€€ђђЦ£
A__inference_dense_layer_call_and_return_conditional_losses_221749^XY0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Џ
™ "&Ґ#
К
0€€€€€€€€€Ў
Ъ {
&__inference_dense_layer_call_fn_221758QXY0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Џ
™ "К€€€€€€€€€Ў°
E__inference_flatten_1_layer_call_and_return_conditional_losses_221704X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ y
*__inference_flatten_1_layer_call_fn_221709K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€©
C__inference_flatten_layer_call_and_return_conditional_losses_221715b8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ "&Ґ#
К
0€€€€€€€€€Ў
Ъ Б
(__inference_flatten_layer_call_fn_221720U8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ "К€€€€€€€€€ЎЉ
D__inference_lambda_1_layer_call_and_return_conditional_losses_222067tAҐ>
7Ґ4
*К'
inputs€€€€€€€€€ђђ

 
p 
™ "/Ґ,
%К"
0€€€€€€€€€ђђ
Ъ Љ
D__inference_lambda_1_layer_call_and_return_conditional_losses_222075tAҐ>
7Ґ4
*К'
inputs€€€€€€€€€ђђ

 
p
™ "/Ґ,
%К"
0€€€€€€€€€ђђ
Ъ Ф
)__inference_lambda_1_layer_call_fn_222080gAҐ>
7Ґ4
*К'
inputs€€€€€€€€€ђђ

 
p 
™ ""К€€€€€€€€€ђђФ
)__inference_lambda_1_layer_call_fn_222085gAҐ>
7Ґ4
*К'
inputs€€€€€€€€€ђђ

 
p
™ ""К€€€€€€€€€ђђЇ
B__inference_lambda_layer_call_and_return_conditional_losses_222041tAҐ>
7Ґ4
*К'
inputs€€€€€€€€€ђђ

 
p 
™ "/Ґ,
%К"
0€€€€€€€€€ђђ
Ъ Ї
B__inference_lambda_layer_call_and_return_conditional_losses_222049tAҐ>
7Ґ4
*К'
inputs€€€€€€€€€ђђ

 
p
™ "/Ґ,
%К"
0€€€€€€€€€ђђ
Ъ Т
'__inference_lambda_layer_call_fn_222054gAҐ>
7Ґ4
*К'
inputs€€€€€€€€€ђђ

 
p 
™ ""К€€€€€€€€€ђђТ
'__inference_lambda_layer_call_fn_222059gAҐ>
7Ґ4
*К'
inputs€€€€€€€€€ђђ

 
p
™ ""К€€€€€€€€€ђђп
I__inference_masking_layer_layer_call_and_return_conditional_losses_222027°nҐk
dҐa
_Ъ\
,К)
inputs/0€€€€€€€€€ђђ
,К)
inputs/1€€€€€€€€€ђђ
™ "/Ґ,
%К"
0€€€€€€€€€ђђ
Ъ «
.__inference_masking_layer_layer_call_fn_222033ФnҐk
dҐa
_Ъ\
,К)
inputs/0€€€€€€€€€ђђ
,К)
inputs/1€€€€€€€€€ђђ
™ ""К€€€€€€€€€ђђо
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_221593ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ є
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_221598j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€<<Ц
™ ".Ґ+
$К!
0€€€€€€€€€Ц
Ъ ∆
0__inference_max_pooling2d_1_layer_call_fn_221603СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€С
0__inference_max_pooling2d_1_layer_call_fn_221608]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€<<Ц
™ "!К€€€€€€€€€Цо
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_221638ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ є
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_221643j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ ".Ґ+
$К!
0€€€€€€€€€Ц
Ъ ∆
0__inference_max_pooling2d_2_layer_call_fn_221648СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€С
0__inference_max_pooling2d_2_layer_call_fn_221653]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ "!К€€€€€€€€€Цо
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_221683ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ є
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_221688j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ ".Ґ+
$К!
0€€€€€€€€€Ц
Ъ ∆
0__inference_max_pooling2d_3_layer_call_fn_221693СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€С
0__inference_max_pooling2d_3_layer_call_fn_221698]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ "!К€€€€€€€€€Цм
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_221548ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ є
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_221553l:Ґ7
0Ґ-
+К(
inputs€€€€€€€€€ђђЦ
™ ".Ґ+
$К!
0€€€€€€€€€<<Ц
Ъ ƒ
.__inference_max_pooling2d_layer_call_fn_221558СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€С
.__inference_max_pooling2d_layer_call_fn_221563_:Ґ7
0Ґ-
+К(
inputs€€€€€€€€€ђђЦ
™ "!К€€€€€€€€€<<Ц≠
A__inference_model_layer_call_and_return_conditional_losses_220948з$%./89BCXYbclmvwАБЛМjҐg
`Ґ]
SЪP
+К(
input_1€€€€€€€€€ђђ
!К
input_2€€€€€€€€€
p 

 
™ "_Ґ\
UЪR
'К$
0/0€€€€€€€€€ђђ
'К$
0/1€€€€€€€€€ђђ
Ъ ≠
A__inference_model_layer_call_and_return_conditional_losses_221022з$%./89BCXYbclmvwАБЛМjҐg
`Ґ]
SЪP
+К(
input_1€€€€€€€€€ђђ
!К
input_2€€€€€€€€€
p

 
™ "_Ґ\
UЪR
'К$
0/0€€€€€€€€€ђђ
'К$
0/1€€€€€€€€€ђђ
Ъ ѓ
A__inference_model_layer_call_and_return_conditional_losses_221249й$%./89BCXYbclmvwАБЛМlҐi
bҐ_
UЪR
,К)
inputs/0€€€€€€€€€ђђ
"К
inputs/1€€€€€€€€€
p 

 
™ "_Ґ\
UЪR
'К$
0/0€€€€€€€€€ђђ
'К$
0/1€€€€€€€€€ђђ
Ъ ѓ
A__inference_model_layer_call_and_return_conditional_losses_221422й$%./89BCXYbclmvwАБЛМlҐi
bҐ_
UЪR
,К)
inputs/0€€€€€€€€€ђђ
"К
inputs/1€€€€€€€€€
p

 
™ "_Ґ\
UЪR
'К$
0/0€€€€€€€€€ђђ
'К$
0/1€€€€€€€€€ђђ
Ъ Д
&__inference_model_layer_call_fn_220446ў$%./89BCXYbclmvwАБЛМjҐg
`Ґ]
SЪP
+К(
input_1€€€€€€€€€ђђ
!К
input_2€€€€€€€€€
p 

 
™ "QЪN
%К"
0€€€€€€€€€ђђ
%К"
1€€€€€€€€€ђђД
&__inference_model_layer_call_fn_220874ў$%./89BCXYbclmvwАБЛМjҐg
`Ґ]
SЪP
+К(
input_1€€€€€€€€€ђђ
!К
input_2€€€€€€€€€
p

 
™ "QЪN
%К"
0€€€€€€€€€ђђ
%К"
1€€€€€€€€€ђђЖ
&__inference_model_layer_call_fn_221470џ$%./89BCXYbclmvwАБЛМlҐi
bҐ_
UЪR
,К)
inputs/0€€€€€€€€€ђђ
"К
inputs/1€€€€€€€€€
p 

 
™ "QЪN
%К"
0€€€€€€€€€ђђ
%К"
1€€€€€€€€€ђђЖ
&__inference_model_layer_call_fn_221518џ$%./89BCXYbclmvwАБЛМlҐi
bҐ_
UЪR
,К)
inputs/0€€€€€€€€€ђђ
"К
inputs/1€€€€€€€€€
p

 
™ "QЪN
%К"
0€€€€€€€€€ђђ
%К"
1€€€€€€€€€ђђ©
C__inference_reshape_layer_call_and_return_conditional_losses_221772b0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Ў
™ ".Ґ+
$К!
0€€€€€€€€€Ц
Ъ Б
(__inference_reshape_layer_call_fn_221777U0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Ў
™ "!К€€€€€€€€€Ц≠
$__inference_signature_wrapper_221076Д$%./89BCXYbclmvwАБЛМsҐp
Ґ 
i™f
6
input_1+К(
input_1€€€€€€€€€ђђ
,
input_2!К
input_2€€€€€€€€€"s™p
4
lambda*К'
lambda€€€€€€€€€ђђ
8
lambda_1,К)
lambda_1€€€€€€€€€ђђо
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_221869ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ є
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_221877j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ ".Ґ+
$К!
0€€€€€€€€€Ц
Ъ ∆
0__inference_up_sampling2d_1_layer_call_fn_221882СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€С
0__inference_up_sampling2d_1_layer_call_fn_221887]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ "!К€€€€€€€€€Цо
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_221924ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ є
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_221932j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ ".Ґ+
$К!
0€€€€€€€€€<<Ц
Ъ ∆
0__inference_up_sampling2d_2_layer_call_fn_221937СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€С
0__inference_up_sampling2d_2_layer_call_fn_221942]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ "!К€€€€€€€€€<<Цо
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_221979ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ї
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_221987l8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€<<Ц
™ "0Ґ-
&К#
0€€€€€€€€€ђђЦ
Ъ ∆
0__inference_up_sampling2d_3_layer_call_fn_221992СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€У
0__inference_up_sampling2d_3_layer_call_fn_221997_8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€<<Ц
™ "#К €€€€€€€€€ђђЦм
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_221814ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ј
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_221822j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ ".Ґ+
$К!
0€€€€€€€€€Ц
Ъ ƒ
.__inference_up_sampling2d_layer_call_fn_221827СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€П
.__inference_up_sampling2d_layer_call_fn_221832]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€Ц
™ "!К€€€€€€€€€Ц