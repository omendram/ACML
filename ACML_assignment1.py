import numpy
import random
import math

##
## BACKPROPAGATION ALGORITHM IMPLEMENTATION
##

def sigmoid_function(x):
	return 1 / (1 + math.exp(-x))

# Formulate inputs 
inputs = []

for x in range(8):
	temp = [1]
	for y in range(8):
		if x == y:
			temp.append(1)
		else:
			temp.append(0)
	inputs.append(temp)

first_layer = []
hidden_layer = [1]
output_layer = []
learning_rate = 0.001

lambda_value = 0.00001
weights_l1_l2 = []
weights_l2_l3 = []


for x in range(9):
	temp_1 = []
	for y in range(3):
		temp_1.append(random.uniform(0, 0.0001))
	weights_l1_l2.append(temp_1)

for y in range(4):
	temp_1 = []
	for x in range(8):
		temp_1.append(random.uniform(0, 0.0001))
	weights_l2_l3.append(temp_1)


def backpropagation(input, wl1, wl2):
	# Feed Forward
	# hidden layer
	h = [sigmoid_function(x) for x in numpy.matmul(input, wl1).tolist()]

	# Output layer
	o = [sigmoid_function(x) for x in numpy.matmul([1] + h, wl2).tolist()]
	
	delta = [-1*(y - h_)*h_*(1 - h_) for y,h_ in zip(input[1:],o)]
	#for y,h_ in zip(input[1:],o): print(y,h_,[-1*(y - h_)*h_*(1 - h_)])
	#print()
        
	delta_2 = numpy.matmul(numpy.array(wl2), numpy.array(delta).transpose())
	delta_2 = [x*h_*(1-h_) for x,h_ in zip(delta_2,h)]

	## Partial derivatives second layer
	##print(numpy.array(delta))
	p_d_w = numpy.matmul(numpy.array(delta).reshape(8, 1), numpy.array(h).reshape(1, 3))
	p_d_b = delta
	
	##print(numpy.array(delta).reshape(8, 1))

	## Partial derivatives first layer
	p_d_w_first_layer = numpy.matmul(numpy.array(delta_2).reshape(3, 1), numpy.array(input).reshape(1, 9))
	p_d_b_first_layer = delta_2
	#print(p_d_w_first_layer)

	return p_d_w, p_d_b, p_d_w_first_layer, p_d_b_first_layer, o;

def calculate_error(output, expected):
    y = [1/2*(op-ex)*(op-ex) for op,ex in zip(output, expected[1:])]
    #for op,ex in zip(output, expected[1:]): print(op,ex,1/2*(op-ex)*(op-ex))
    #print(numpy.sum(y))
    return numpy.sum(y)

def sum_all_weights(wl1, wl2):
	y1 = numpy.sum([numpy.sum(numpy.square(x)) for x in wl2[1:]])
	y2 = numpy.sum([numpy.sum(numpy.square(x)) for x in wl1[1:]])

	return (y1 + y2);


def param_update(wl1, wl2, inp):
	delta_w = numpy.array([0 for i in range(24)]).reshape(8, 3)
	delta_b = numpy.array([0 for i in range(8)]).reshape(1, 8)
	delta_w_first_layer = numpy.array([0 for i in range(27)]).reshape(9,3)
	delta_b_first_layer = numpy.array([0 for i in range(3)]).reshape(1,3)
	J=0

	for input in inp:
		p_d_w, p_d_b, p_d_w_first_layer, p_d_b_first_layer, o = backpropagation(input, wl1, wl2)
		J = J + calculate_error(o, input)
		#print(delta_w_first_layer)
		#print(numpy.array(p_d_w_first_layer).transpose())
	#	print(input ,o)
		delta_w = delta_w + p_d_w
		delta_b = delta_b + p_d_b
		delta_w_first_layer = delta_w_first_layer + numpy.array(p_d_w_first_layer).transpose()
		delta_b_first_layer = delta_b_first_layer + numpy.array(p_d_b_first_layer).transpose()
		#print(delta_w_first_layer)
	#	print()
	

	J = J / len(inp) + (lambda_value / 2) * sum_all_weights(wl1, wl2)

	wl1[1:] = wl1[1:] - learning_rate*(1/len(inp)*delta_w_first_layer[1:] + lambda_value*numpy.array(wl1[1:]))
	wl2[1:] = wl2[1:] - learning_rate*(1/len(inp)*delta_w.transpose() + lambda_value*numpy.array(wl2[1:]))
	wl1[0] = wl1[0] - learning_rate*(1 / len(inp) * numpy.array(p_d_b_first_layer))
	wl2[0] = wl2[0] - learning_rate*(1 / len(inp) * numpy.array(p_d_b))

	return wl1, wl2, J



def gradient_descent(wl1, wl2, inp):
    count = 0
    counter = 0
    inp = inp[:7]
 
    # training
    while(count != len(inp)):
            counter = counter +1
            count = 0
            wl1, wl2, J = param_update(wl1, wl2, inp)
            for input in inp:
                A, B, C, D, o = backpropagation(input, wl1, wl2)
                if input[1:].index(max(input[1:])) == o.index(max(o)):
                    count = count + 1
                if counter%25000 == 0:
                    print(input[1:].index(max(input[1:])),o.index(max(o)), count)
                    

    for input in inp:
        A, B, C, D, o = backpropagation(input, wl1, wl2)
        if input[1:].index(max(input[1:])) == o.index(max(o)):
            count = count + 1
            print(input[1:].index(max(input[1:])),input[1:].index(max(input[1:])), max(o))
            print(o)

    print(counter)

    for input in inp:
        A, B, C, D, o = backpropagation(input, wl1, wl2)
        print(input[1:])
        print(o)
                
    
    
    print(count)
                
                
            

gradient_descent(weights_l1_l2, weights_l2_l3, inputs)

print(weights_l1_l2)
print(weights_l2_l3)
