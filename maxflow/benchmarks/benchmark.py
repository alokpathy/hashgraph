#!/usr/bin/python
import os
import sys
import subprocess
import random
import time
import argparse


#config

#maxflow implementations
implementations = [ "mpm-gpu-naive", "galois-preflowpush"]
log_filename = "bench_log.csv"

#end config

#parsing args
parser = argparse.ArgumentParser(description="Compute benchmarks of maxflow implementations")
parser.add_argument("--log", dest='log', action='store_const', const=1, default=0, help="Save individual benchmark results to logfile")
parser.add_argument("--make", dest='make', action='store_const', const=1, default=0, help="Make maxflow implementations")

args = parser.parse_args()
log = args.log
make = args.make

#make executables
for implementation in implementations:
	if make:
		subprocess.call(["make", "-C", "..", "clean"])
		subprocess.call(["make", "-C", "..", implementation, "LOG_LEVEL=1"])
	else: 
		if not os.path.isfile("../" + implementation):	
			print("../" + implementation + " does not exist. Please use --make to compile it.")
			sys.exit(1)


commit_hash = subprocess.Popen(["git", "log", "-n", "1", "--pretty=format:\"%h\""], stdout=subprocess.PIPE).communicate()[0]
commit_title = subprocess.Popen(["git", "log", "-n", "1", "--pretty=format:\"%s\""], stdout=subprocess.PIPE).communicate()[0]
time_bench = time.time()

if log:
	logfile = open(log_filename, "a")
	

#text coloring
def colorRef(val):
	return "\033[94m" + str(val) + "\033[0m"


def colorPassed(val):
	return "\033[92m" + str(val) + "\033[0m"


def colorFailed(val):
	return "\033[91m" + str(val) + "\033[0m"


def argmin(lst):
	return lst.index(min(lst))

def argsort(seq):
    return [x for x,y in sorted(enumerate(seq), key = lambda x: x[1])]

#extract runtime and flow from program output
def flow_time_extract(res):
	time = res.rsplit(None, 2)[-2].rstrip()
	flow = res.rsplit(None, 4)[-4].rstrip()
	return flow,time	

def test(matrix, s, t, w=None):
	global failed,passed,winners
	filename, file_extension = os.path.splitext(matrix)
	
	matrix = "../" + matrix

	if file_extension != '.gr': #we need ton convert the graph first
		if not os.path.isfile(matrix + '.gr'):
			print "Converting " + matrix + " to gr format..."
			subprocess.call(['../data/convert_graph.sh', matrix, matrix + '.gr'])
		
	matrix += '.gr'	
	
	times = []
	out_line =  [matrix, str(s), str(t)]

	for i in range(len(implementations)):

		implementation = implementations[i]
	
		res = subprocess.Popen(["../" + implementation, matrix, str(s), str(t)], stdout=subprocess.PIPE).communicate()[0]
		flow,time = flow_time_extract(res)
		
		if i==0: #reference
			ref_flow = flow
			out_line.append(ref_flow)
			times.append(float(time))
			out_line.append(colorRef(time))
		else:
			if flow == ref_flow:
				out_line.append(colorPassed(time))
				times.append(float(time))
				passed += 1
			else:
				out_line.append(colorFailed(time))
				times.append(sys.maxint)
				failed += 1
		if log:
  			logfile_line = [str(time_bench), commit_hash, commit_title, implementation, matrix, str(s), str(t), time, flow]
			logfile.write(', '.join(logfile_line) + "\n")
			logfile.flush()
	
	best = argmin(times)
	winners[best] += 1
	
	out_line.append(implementations[best])
		
	
	print ', '.join(out_line)	

passed = 0
failed = 0

#winners[i] : number of times implementations[i] was the best one
winners = [0] * len(implementations)

print '=== BENCHMARKS ==='

random.seed(1234321)

# save header
header = ['matrix', 'source', 'sink', 'flow']
header.extend(implementations)
header.append("best")
print ', '.join(header)	

test('data/wiki2003.mtx', 3, 12563)
test('data/wiki2003.mtx', 54, 1432)
test('data/wiki2003.mtx', 65, 7889)
test('data/wiki2003.mtx', 43242, 5634)
test('data/wiki2003.mtx', 78125, 327941)
test('data/wiki2003.mtx', 2314, 76204)

test('data/roadNet-CA.mtx', 2314, 76204)
test('data/roadNet-CA.mtx', 9, 1247)
test('data/roadNet-CA.mtx', 1548, 365940)
test('data/roadNet-CA.mtx', 1548785, 654123)
test('data/roadNet-CA.mtx', 8, 284672)

# USA road network (23.9M vertices, 28.8M edges)
test('data/road_usa.mtx', 125, 7846232)
test('data/road_usa.mtx', 458743, 321975)
test('data/road_usa.mtx', 96, 4105465)
test('data/road_usa.mtx', 5478, 658413)
test('data/road_usa.mtx', 364782, 32)
#test('data/road_usa.mtx', 21257849, 2502578)
#test('data/road_usa.mtx', 12345678, 23000000)
#test('data/road_usa.mtx', 16807742, 17453608)

# wikipedia (3.7M vertices, 66.4M edges) 
test('data/wiki2011.mtx', 254, 87452)
test('data/wiki2011.mtx', 315547, 874528)
test('data/wiki2011.mtx', 8796, 673214)

if log:
	logfile.close()

print '=== SUMMARY ==='
print str(failed) + ' tests failed out of ' + str(passed + failed)
print "Implementations ranking : "
w_indexes = reversed(argsort(winners))
for w_idx in w_indexes:
	print implementations[w_idx] + " : " + str(winners[w_idx]) + " win(s)"


