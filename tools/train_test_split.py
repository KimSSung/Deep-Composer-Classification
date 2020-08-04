# PLEASE run in 'split' folder


import os
import random

INPUT_PATH = '/data/inputs/'
composer = []
composer_seg_count = [] # Total segment num of each composer
composer_midi_count = [] # Total midi folder num of each composer
seg_per_midi = [] # len = # of composer. # of seg per each midi

train_percentage = 0.7

random.seed(333) # change this

train_file = open('/data/split/train.txt', 'w')
test_file = open('/data/split/test.txt', 'w')

for folder in os.listdir(INPUT_PATH):
	if folder == 'name_id_map.csv': continue
	composer.append(int(folder.replace('composer', '')))

	composer_fold = INPUT_PATH + folder + '/'
	count = 0
	midi_count = 0
	seg_per_midi_composer = []
	for midi_f in os.listdir(composer_fold):
		# print(midi_f)
		midi_count += 1
		midi_fold = composer_fold + midi_f
		seg_count = 0 # segment count of each midi
		for file in os.listdir(midi_fold):
			count += 1
			seg_count += 1

		seg_per_midi_composer.append(seg_count)
	seg_per_midi.append(seg_per_midi_composer)



	# print(count)
	composer_seg_count.append(count)
	composer_midi_count.append(midi_count)

print("## composer index:")
print(composer)
print("## composer's total seg counts:")
print(composer_seg_count)
print("## composer's total midi counts:")
print(composer_midi_count)
# print("## seg per midi counts:")
# print(seg_per_midi)
print()

###########################################################################
###########################################################################
# # 1. Split per Segment
# train_idx = []
# for count in composer_seg_count:
# 	# print(list(range(0, count)))

# 	idxlist = random.sample(list(range(0, count)), int(count * train_percentage))
# 	train_idx.append(idxlist)


# total_train_count = 0
# total_test_count = 0
# composer_idx = 0 # to exclude 'csv' file on composer_idx count
# for folder in os.listdir(INPUT_PATH):
# 	if folder == 'name_id_map.csv': continue
# 	composer_fold = INPUT_PATH + folder + '/'
# 	idx = -1
# 	for midi_f in os.listdir(composer_fold):
# 		# print(midi_f)
# 		midi_fold = composer_fold + midi_f + '/'
# 		for file in os.listdir(midi_fold):
# 			idx += 1

# 			if idx in train_idx[composer_idx]:
# 				# print(midi_fold + file)
# 				train_file.write(midi_fold + file + '\n')
# 				total_train_count += 1
				
# 			else:
# 				test_file.write(midi_fold + file + '\n')
# 				total_test_count += 1

#	composer_idx += 1
				


# print('## After split:')
# print("total:", sum(composer_seg_count))
# print("train num:", int(sum(composer_seg_count) * train_percentage))
# print("test num:", int(sum(composer_seg_count) * (1 - train_percentage)))
# print()

# print('total train count:', total_train_count)
# print('total test count:', total_test_count)


###########################################################################
###########################################################################
## 2. Split per Midi
seg_idxs = []
midi_idxs = []

# Select 10 segments from each midi
for composer in seg_per_midi:
	seg_idx_composer = []
	for seg_count in composer:
		idxlist = random.sample(list(range(0, seg_count)), 10)
		seg_idx_composer.append(idxlist)
	seg_idxs.append(seg_idx_composer)
# Select train / test 'midi' from each composer
for midi_count in composer_midi_count:
	idxlist = random.sample(list(range(0, midi_count)), int(midi_count*train_percentage))
	midi_idxs.append(idxlist)


total_train_count = 0
total_test_count = 0
composer_idx = 0 # to exclude 'csv' file on composer_idx count
for folder in os.listdir(INPUT_PATH):
	if folder == 'name_id_map.csv': continue
	# print(int(folder.replace('composer', '')))
	
	composer_fold = INPUT_PATH + folder + '/'
	for this_midi_idx, midi_f in enumerate(os.listdir(composer_fold)):
		
		midi_fold = composer_fold + midi_f + '/'
		for this_seg_idx, file in enumerate(os.listdir(midi_fold)):

			# Train
			if this_midi_idx in midi_idxs[composer_idx]:
				if this_seg_idx in seg_idxs[composer_idx][this_midi_idx]: # in 10 Seg list
					train_file.write(midi_fold + file + '\n')
					total_train_count += 1
			# Test
			else:
				if this_seg_idx in seg_idxs[composer_idx][this_midi_idx]: # in 10 Seg list
					test_file.write(midi_fold + file + '\n')
					total_test_count += 1

	composer_idx += 1


print('## After split:')
print('total train count:', total_train_count)
print('total test count:', total_test_count)


train_file.close()
test_file.close()