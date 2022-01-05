from random import gauss
from csv import writer
import sys

with open(sys.argv[1], 'w') as csvfile:
	wrt = writer(csvfile, delimiter=',')
	for i in range(int(sys.argv[2])):
		wrt.writerow([str(gauss(0,10)) for _ in range(int(sys.argv[3]))])
