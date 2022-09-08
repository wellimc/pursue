import sys
import os
import subprocess
import numpy as np


RUN_SCRIPT = 'run_video.py'
RANDOM_SEED = 42
RUN_TIME = 280  # sec
#ABR_ALGO = ['fastMPC', 'robustMPC', 'BOLA', 'RL']
ABR_ALGO = ['RL']
REPEAT_TIME = 1


def main():

	np.random.seed(RANDOM_SEED)

	with open('./chrome_retry_log', 'w') as log:
		log.write('chrome retry log\n')
		log.flush()

		for rt in range(REPEAT_TIME):
			#np.random.shuffle(ABR_ALGO)
			#for abr_algo in ABR_ALGO:

			#	while True:

					script = 'python3 ' + RUN_SCRIPT + ' ' + \
							  'RL''' + ' ' + str(RUN_TIME) + ' ' + str(rt)
					
					print(script)
					proc = subprocess.Popen(script,
							  stdout=subprocess.PIPE, 
							  stderr=subprocess.PIPE, 
							  shell=True)

					(out, err) = proc.communicate()

					if out == 'done\n':
						break
					else:
						log.write(str('RL') + '_' + str(rt) + '\n')
						log.write(str(out) + '\n')
						log.write(str(err) + '\n')
						log.flush()



if __name__ == '__main__':
	main()
