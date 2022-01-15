from random import gauss
from csv import writer
import sys
from tqdm import tqdm

with open(sys.argv[1], 'w') as csvfile:
	wrt = writer(csvfile, delimiter=',')
	for i in tqdm(range(int(sys.argv[2]))):
		wrt.writerow([str(gauss(0,10)) for _ in range(int(sys.argv[3]))])
