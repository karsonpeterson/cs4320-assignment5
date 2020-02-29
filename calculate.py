import scipy.special

def calculate(num_models, model_correctness):
	majority_count = 1 + int(num_models / 2)
	p = 0.0
	for i in range(majority_count, num_models+1):
		count = scipy.special.comb(num_models, i)
		p += count * (model_correctness**i) * ((1.0-model_correctness) ** (num_models-i))

	return p

def main():
	num_models = 1000
	model_correctness = 0.51
	majority_correct_probability = calculate(num_models, model_correctness)
	print(num_models, 'models at', model_correctness, 'will have majority correct with probability', majority_correct_probability)
	return

if __name__ == '__main__':
	main()