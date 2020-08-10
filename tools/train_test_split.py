# PLEASE run in 'split' folder


import os
import random

# change this
# each_seg = 20
# INPUT_PATH = '/data/inputs_overlap/seg_' + str(each_seg) + '/'
INPUT_PATH = '/data/inputs_transpose/test/'


composer = []
composer_seg_count = [] # Total segment num of each composer
composer_midi_count = [] # Total midi folder num of each composer
seg_per_midi = [] # len = # of composer. # of seg per each midi

total_augfold_cnt = 0 # total aug fold num

train_percentage = 0.7
print("###################################")
print(">> Train : Test = " + str(int(train_percentage*10)) + " : " + str(int((1 - train_percentage)*10)) + " <<")
print()
print()

random.seed(333) # change this

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
		midi_fold = composer_fold + midi_f + '/'
		
	# 	# (1) Before augmentation: INPUT_PATH == '/data/inputs/'
	# 	seg_count = 0 # segment count of each midi
	# 	for file in os.listdir(midi_fold): 
	# 		count += 1
	# 		seg_count += 1
	# 	seg_per_midi_composer.append(seg_count)
	# seg_per_midi.append(seg_per_midi_composer)


		# (2) After augmentation: INPUT_PATH == '/data/inputs_transpose/'
		aug_per_midi = []
		for midi_augf in os.listdir(midi_fold): # aug folder
			aug_path = midi_fold + midi_augf + '/'
			total_augfold_cnt += 1

			seg_count = 0
			for file in os.listdir(aug_path):
				count += 1
				seg_count += 1

			aug_per_midi.append(seg_count)
		seg_per_midi_composer.append(aug_per_midi)
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

# min_cnt = 30000
# for comp_list in seg_per_midi: # 13 composer
# 	for midi_list in comp_list:
# 		for cnt in midi_list:
# 			if min_cnt > cnt:
# 				min_cnt = cnt

# print("## Min seg count:")
# print(min_cnt)
# print("## Total augmentation num:")
# print(total_augfold_cnt*min_cnt)
# print()


# Consider Age
# 1. Baroque: Scarlatti / Bach => [2, 6]
# 2. Classical: Haydn / Mozart / Beethoven / Schubert => [4, 8, 9, 12]
# 3. Romanticism: Schumann / Chopin / Liszt / Brahms / Debussy
#                 / Rachmaninoff / Scriabin => [0, 1, 3, 5, 7, 10, 11]

baroq_seg, classic_seg, roman_seg = [], [], []
baroq_midi, classic_midi, roman_midi = [], [], []

for i in range(13):
	if i == 2 or i == 6:
		baroq_seg.append(composer_seg_count[i])
		baroq_midi.append(composer_midi_count[i])
	elif i == 4 or i == 8 or i == 9 or i == 12:
		classic_seg.append(composer_seg_count[i])
		classic_midi.append(composer_midi_count[i])
	else:
		roman_seg.append(composer_seg_count[i])
		roman_midi.append(composer_midi_count[i])		

print("Baroque:")
print("Seg:", sum(baroq_seg))
print("Midi:", sum(baroq_midi))
print()
print("Classical:")
print("Seg:", sum(classic_seg))
print("Midi:", sum(classic_midi))
print()
print("Romanticism:")
print("Seg:", sum(roman_seg))
print("Midi:", sum(roman_midi))
print()

# each_seg = min_cnt
each_seg = 10


train_file = open('/data/split/train.txt', 'w')
test_file = open('/data/split/test.txt', 'w')

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

##############################################
# Select 'each seg' segments from each midi
# (1) Before aug
# for composer in seg_per_midi:
# 	seg_idx_composer = []
# 	for seg_count in composer:
# 		idxlist = random.sample(list(range(0, seg_count)), each_seg)
# 		seg_idx_composer.append(idxlist)
# 	seg_idxs.append(seg_idx_composer)

# (2) After aug
for comp_list in seg_per_midi: # 13 composer
	seg_of_midi = []
	for midi_list in comp_list:
		seg_idx = []
		for seg_count in midi_list:
			idxlist = random.sample(list(range(0, seg_count)), each_seg)
			seg_idx.append(idxlist)
		seg_of_midi.append(seg_idx)
	seg_idxs.append(seg_of_midi)

# print(seg_idxs)


##############################################

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

		##############################################
		# 1. Before aug
		# for this_seg_idx, file in enumerate(os.listdir(midi_fold)):

		# 	# Train
		# 	if this_midi_idx in midi_idxs[composer_idx]:
		# 		if this_seg_idx in seg_idxs[composer_idx][this_midi_idx]: # in each_seg Seg list
		# 			train_file.write(midi_fold + file + '\n')
		# 			total_train_count += 1
		# 	# Test
		# 	else:
		# 		if this_seg_idx in seg_idxs[composer_idx][this_midi_idx]: # in each_seg Seg list
		# 			test_file.write(midi_fold + file + '\n')
		# 			total_test_count += 1
		##############################################


		##############################################
		# 2. After aug
		for aug_idx, aug_f in enumerate(os.listdir(midi_fold)):
			aug_fold = midi_fold + aug_f + '/'

			for this_seg_idx, file in enumerate(os.listdir(aug_fold)):

				# print(seg_idxs[composer_idx][this_midi_idx][aug_idx])

				# Train
				if this_midi_idx in midi_idxs[composer_idx]:
					if this_seg_idx in seg_idxs[composer_idx][this_midi_idx][aug_idx]: # in each_seg Seg list
						train_file.write(aug_fold + file + '\n')
						total_train_count += 1
				# Test
				else:
					if this_seg_idx in seg_idxs[composer_idx][this_midi_idx][aug_idx]: # in each_seg Seg list
						test_file.write(aug_fold + file + '\n')
						total_test_count += 1

		##############################################


	composer_idx += 1


print('## After split:')
print('total train count:', total_train_count)
print('total test count:', total_test_count)


train_file.close()
test_file.close()


